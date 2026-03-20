#!/usr/bin/env python3
"""Correctness test for lj_force (Real-World) - attempt 2"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from realworld_results.llm_triton.lj_force.attempt2 import lj_force_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "liblj_force.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(f_c, neighbors_c, numneigh_c, pos_c, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_neighbors = ctypes.c_int * (512000)
    c_arr_neighbors = CType_neighbors.in_dll(lib, 'neighbors')
    src_neighbors = np.ascontiguousarray(neighbors_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_neighbors, src_neighbors.ctypes.data, src_neighbors.nbytes)
    CType_numneigh = ctypes.c_int * (4000)
    c_arr_numneigh = CType_numneigh.in_dll(lib, 'numneigh')
    src_numneigh = np.ascontiguousarray(numneigh_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_numneigh, src_numneigh.ctypes.data, src_numneigh.nbytes)
    CType_pos = ctypes.c_float * (12000)
    c_arr_pos = CType_pos.in_dll(lib, 'pos')
    src_pos = np.ascontiguousarray(pos_c, dtype=np.float32)
    ctypes.memmove(c_arr_pos, src_pos.ctypes.data, src_pos.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'cutforcesq_val').value = float(cutforcesq_val)
    ctypes.c_float.in_dll(lib, 'epsilon_val').value = float(epsilon_val)
    ctypes.c_float.in_dll(lib, 'sigma6_val').value = float(sigma6_val)

    # Run kernel
    func = getattr(lib, "lj_force_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_f = ctypes.c_float * (12000)
    c_arr_f = CType_f.in_dll(lib, 'f')
    f_c[:] = np.frombuffer(c_arr_f, dtype=np.float32).reshape(12000).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            NLOCAL = 4000
            MAXNEIGHS = 128
            PAD = 3
            # Atom positions on a lattice in box [0, L]^3, spaced for ~60 neighbors
            # Lattice spacing 2.0, cutoff 4.0 → each atom sees ~60 neighbors
            import numpy as np_init
            lattice_n = int(round(NLOCAL ** (1.0/3.0))) + 1
            spacing = 2.0
            coords = []
            for ix in range(lattice_n):
                for iy in range(lattice_n):
                    for iz in range(lattice_n):
                        if len(coords) >= NLOCAL:
                            break
                        # Small random perturbation
                        coords.append([ix*spacing + np_init.random.uniform(-0.2, 0.2),
                                       iy*spacing + np_init.random.uniform(-0.2, 0.2),
                                       iz*spacing + np_init.random.uniform(-0.2, 0.2)])
                    if len(coords) >= NLOCAL:
                        break
                if len(coords) >= NLOCAL:
                    break
            coords = np_init.array(coords[:NLOCAL], dtype=np_init.float32)
            pos = torch.from_numpy(coords.flatten()).to('cuda')
            f = torch.zeros(NLOCAL * PAD, device='cuda', dtype=torch.float32)
            # Build neighbor lists using scipy KDTree for efficiency
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            cutoff = 4.0
            numneigh_list = []
            neighbors_flat = []
            for i_atom in range(NLOCAL):
                nlist = tree.query_ball_point(coords[i_atom], cutoff)
                nlist = [j for j in nlist if j != i_atom]
                if len(nlist) > MAXNEIGHS:
                    nlist = nlist[:MAXNEIGHS]
                numneigh_list.append(len(nlist))
                padded = nlist + [0] * (MAXNEIGHS - len(nlist))
                neighbors_flat.extend(padded)
            numneigh = torch.tensor(numneigh_list, device='cuda', dtype=torch.int32)
            neighbors = torch.tensor(neighbors_flat, device='cuda', dtype=torch.int32)
            cutforcesq_val = 16.0  # cutoff^2
            sigma6_val = 1.0
            epsilon_val = 1.0
            cutforcesq_val = 1.0
            epsilon_val = 1.0
            sigma6_val = 1.0
            MAXNEIGHS = 128
            NLOCAL = 4000
            PAD = 3

            # Clone for C reference
            f_c = f.cpu().numpy().copy()
            neighbors_c = neighbors.cpu().numpy().copy()
            numneigh_c = numneigh.cpu().numpy().copy()
            pos_c = pos.cpu().numpy().copy()

            # Clone for Triton
            f_tr = f.clone()
            neighbors_tr = neighbors.clone()
            numneigh_tr = numneigh.clone()
            pos_tr = pos.clone()

            # Run C reference
            run_c_reference(f_c, neighbors_c, numneigh_c, pos_c, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)

            # Run Triton
            lj_force_triton(f_tr, neighbors_tr, numneigh_tr, pos_tr, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(f_c).float()
            tr_val = f_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

            # Pass if absolute error < atol OR relative error < rtol
            passed = (max_error < 0.001) or (max_rel_error < 0.001)
            if passed:
                print(f"  Test {test_idx + 1}: PASS (abs={max_error:.6e} rel={max_rel_error:.6e})")
            else:
                print(f"  Test {test_idx + 1}: FAIL (abs={max_error:.6e} rel={max_rel_error:.6e})")
                all_passed = False

        except Exception as e:
            print(f"  Test {test_idx + 1}: ERROR - {e}")
            all_passed = False

    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    return all_passed

if __name__ == "__main__":
    test_correctness()

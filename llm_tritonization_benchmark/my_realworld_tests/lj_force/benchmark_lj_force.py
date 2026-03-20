#!/usr/bin/env python3
"""Performance Benchmark for lj_force (Real-World)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from realworld_results.llm_triton.lj_force.attempt2 import lj_force_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "liblj_force.so"

def run_c_reference(f_c, neighbors_c, numneigh_c, pos_c, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD):
    lib = ctypes.CDLL(str(C_LIB_PATH))
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
    ctypes.c_float.in_dll(lib, 'cutforcesq_val').value = float(cutforcesq_val)
    ctypes.c_float.in_dll(lib, 'epsilon_val').value = float(epsilon_val)
    ctypes.c_float.in_dll(lib, 'sigma6_val').value = float(sigma6_val)
    func = getattr(lib, "lj_force_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_f = ctypes.c_float * (12000)
    c_arr_f = CType_f.in_dll(lib, 'f')
    f_c[:] = np.frombuffer(c_arr_f, dtype=np.float32).reshape(12000).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

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

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            f_c = f.cpu().numpy().copy()
            neighbors_c = neighbors.cpu().numpy().copy()
            numneigh_c = numneigh.cpu().numpy().copy()
            pos_c = pos.cpu().numpy().copy()
            run_c_reference(f_c, neighbors_c, numneigh_c, pos_c, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)
        start = time.perf_counter()
        for _ in range(num_iterations):
            f_c = f.cpu().numpy().copy()
            neighbors_c = neighbors.cpu().numpy().copy()
            numneigh_c = numneigh.cpu().numpy().copy()
            pos_c = pos.cpu().numpy().copy()
            run_c_reference(f_c, neighbors_c, numneigh_c, pos_c, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            f_tr = f.clone()
            neighbors_tr = neighbors.clone()
            numneigh_tr = numneigh.clone()
            pos_tr = pos.clone()
            lj_force_triton(f_tr, neighbors_tr, numneigh_tr, pos_tr, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            f_tr = f.clone()
            neighbors_tr = neighbors.clone()
            numneigh_tr = numneigh.clone()
            pos_tr = pos.clone()
            lj_force_triton(f_tr, neighbors_tr, numneigh_tr, pos_tr, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"Triton error: {e}")

    # Report
    speedup = c_time / tr_time if c_time and tr_time and tr_time > 0 else None
    c_ms = c_time * 1000 if c_time else -1
    tr_ms = tr_time * 1000 if tr_time else -1
    sp = speedup if speedup else -1

    print(f"C ref:   {c_ms:8.3f} ms")
    print(f"Triton:  {tr_ms:8.3f} ms")
    if speedup:
        print(f"Speedup: {speedup:8.2f}x")
    else:
        print(f"Speedup: N/A")
    print(f"BENCHMARK_RESULT:{c_ms:.6f},{tr_ms:.6f},{sp:.6f}")

if __name__ == "__main__":
    benchmark()

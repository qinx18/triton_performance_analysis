import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, j_idx, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid_i = tl.program_id(0)
    i = pid_i + 1  # Start from 1
    
    if i >= LEN_2D:
        return
    
    # Load a[i-1] (scalar)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Load aa[j_idx][i] (scalar)
    aa_val = tl.load(aa_ptr + j_idx * LEN_2D + i)
    
    # Load bb[j_idx][i] (scalar)
    bb_val = tl.load(bb_ptr + j_idx * LEN_2D + i)
    
    # Compute a[i] = aa[j_idx][i] - a[i-1]
    a_new = aa_val - a_prev
    
    # Store a[i]
    tl.store(a_ptr + i, a_new)
    
    # Compute aa[j_idx][i] = a[i] + bb[j_idx][i]
    aa_new = a_new + bb_val
    
    # Store aa[j_idx][i]
    tl.store(aa_ptr + j_idx * LEN_2D + i, aa_new)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Sequential over i (1 to LEN_2D-1), parallel over j
    for i in range(1, LEN_2D):
        # For this i, process all j values in parallel
        for j in range(LEN_2D):
            # Load a[i-1]
            a_prev = a[i-1].item()
            
            # Load aa[j][i] and bb[j][i]
            aa_val = aa[j, i].item()
            bb_val = bb[j, i].item()
            
            # Compute a[i] = aa[j][i] - a[i-1]
            a_new = aa_val - a_prev
            a[i] = a_new
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            aa_new = a_new + bb_val
            aa[j, i] = aa_new
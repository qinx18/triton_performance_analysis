import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    # Load a[i-1]
    a_prev = tl.load(a_ptr + i_val - 1)
    
    # For each j, compute a[i] = aa[j][i] - a[i-1]
    aa_col_ptrs = aa_ptr + j_offsets * N + i_val
    bb_col_ptrs = bb_ptr + j_offsets * N + i_val
    
    aa_vals = tl.load(aa_col_ptrs, mask=j_mask)
    bb_vals = tl.load(bb_col_ptrs, mask=j_mask)
    
    # Compute a[i] for each j iteration - last j overwrites
    a_vals = aa_vals - a_prev
    
    # Find the last valid j (highest index)
    valid_indices = tl.where(j_mask, j_offsets, -1)
    last_valid_j = tl.max(valid_indices)
    
    # Get the final a[i] value from the last j iteration
    final_a_val = tl.sum(tl.where(j_offsets == last_valid_j, a_vals, 0.0))
    
    # Store a[i] (only first thread block does this)
    if tl.program_id(0) == 0:
        tl.store(a_ptr + i_val, final_a_val)
    
    # Update aa[j][i] = a[i] + bb[j][i] for all valid j
    new_aa_vals = final_a_val + bb_vals
    tl.store(aa_col_ptrs, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    N = aa.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Sequential loop over i from 1 to N-1
    for i in range(1, N):
        s257_kernel[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)
import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    # Load a[i-1]
    a_prev = tl.load(a_ptr + i_val - 1)
    
    # Load aa[j][i] and bb[j][i] for all j
    aa_ptrs = aa_ptr + j_offsets * N + i_val
    bb_ptrs = bb_ptr + j_offsets * N + i_val
    
    aa_vals = tl.load(aa_ptrs, mask=j_mask)
    bb_vals = tl.load(bb_ptrs, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for each j
    # Since each j overwrites a[i], take the value from last valid j
    a_new_vals = aa_vals - a_prev
    
    # Find the maximum valid j index
    max_valid_j = tl.max(tl.where(j_mask, j_offsets, -1))
    
    # Get a[i] from the highest j that's valid
    final_a_val = tl.sum(tl.where(j_offsets == max_valid_j, a_new_vals, 0.0))
    
    # Store final a[i] value
    if tl.program_id(0) == 0:
        tl.store(a_ptr + i_val, final_a_val)
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    new_aa_vals = final_a_val + bb_vals
    tl.store(aa_ptrs, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    for i in range(1, N):
        s257_kernel[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)
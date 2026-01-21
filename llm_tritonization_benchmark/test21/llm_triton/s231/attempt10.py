import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(aa, bb, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    for j in range(1, N):
        # Load aa[j-1, i] (previous row)
        aa_prev_ptrs = aa + (j - 1) * N + i_idx
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        
        # Load bb[j, i] (current row)
        bb_ptrs = bb + j * N + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j, i] = aa[j-1, i] + bb[j, i]
        result = aa_prev_vals + bb_vals
        
        # Store result to aa[j, i]
        aa_curr_ptrs = aa + j * N + i_idx
        tl.store(aa_curr_ptrs, result, mask=i_mask)

def s231_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s231_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)
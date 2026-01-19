import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        curr_row_ptr = aa_ptr + i * N + j_idx
        prev_row_ptr = aa_ptr + (i - 1) * N + j_idx
        bb_row_ptr = bb_ptr + i * N + j_idx
        
        prev_vals = tl.load(prev_row_ptr, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_row_ptr, mask=j_mask, other=0.0)
        result = prev_vals + bb_vals
        
        tl.store(curr_row_ptr, result, mask=j_mask)

def s1119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, N=N, BLOCK_SIZE=BLOCK_SIZE)
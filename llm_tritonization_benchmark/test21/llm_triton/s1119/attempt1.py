import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        # Current row indices
        curr_row_base = i * N
        prev_row_base = (i - 1) * N
        
        # Load current row data
        curr_idx = curr_row_base + j_idx
        prev_idx = prev_row_base + j_idx
        
        # Load values
        aa_prev = tl.load(aa_ptr + prev_idx, mask=j_mask, other=0.0)
        bb_curr = tl.load(bb_ptr + curr_idx, mask=j_mask, other=0.0)
        
        # Compute
        result = aa_prev + bb_curr
        
        # Store result
        tl.store(aa_ptr + curr_idx, result, mask=j_mask)

def s1119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)
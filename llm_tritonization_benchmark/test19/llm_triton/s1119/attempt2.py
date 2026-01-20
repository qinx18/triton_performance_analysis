import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < n
    
    for i in range(1, n):
        # Load from aa[i-1][j] (previous row)
        prev_row_indices = (i - 1) * n + j_idx
        aa_prev = tl.load(aa_ptr + prev_row_indices, mask=j_mask, other=0.0)
        
        # Load from bb[i][j] (current row)
        curr_row_indices = i * n + j_idx
        bb_curr = tl.load(bb_ptr + curr_row_indices, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
        result = aa_prev + bb_curr
        
        # Store to aa[i][j]
        tl.store(aa_ptr + curr_row_indices, result, mask=j_mask)

def s1119_triton(aa, bb):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1119_kernel[grid](aa, bb, n, BLOCK_SIZE=BLOCK_SIZE)
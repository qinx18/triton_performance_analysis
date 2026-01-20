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
        # Current row indices
        current_row_base = i * n
        current_indices = current_row_base + j_idx
        
        # Previous row indices  
        prev_row_base = (i - 1) * n
        prev_indices = prev_row_base + j_idx
        
        # Load data
        aa_prev = tl.load(aa_ptr + prev_indices, mask=j_mask, other=0.0)
        bb_curr = tl.load(bb_ptr + current_indices, mask=j_mask, other=0.0)
        
        # Compute
        result = aa_prev + bb_curr
        
        # Store
        tl.store(aa_ptr + current_indices, result, mask=j_mask)

def s1119_triton(aa, bb):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1119_kernel[grid](aa, bb, n, BLOCK_SIZE=BLOCK_SIZE)
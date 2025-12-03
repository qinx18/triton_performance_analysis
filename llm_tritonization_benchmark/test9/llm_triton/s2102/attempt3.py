import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Each block processes a range of columns
    col_offsets = block_start + offsets
    col_mask = col_offsets < n
    
    # For each column in this block
    for col_local in range(BLOCK_SIZE):
        col_idx = block_start + col_local
        if col_idx >= n:
            break
            
        # Zero out entire column col_idx
        for row_start in range(0, n, BLOCK_SIZE):
            row_offsets = row_start + offsets
            row_mask = row_offsets < n
            
            # Calculate memory addresses for aa[row_offsets][col_idx]
            mem_offsets = row_offsets * n + col_idx
            tl.store(aa_ptr + mem_offsets, 0.0, mask=row_mask)
        
        # Set diagonal element aa[col_idx][col_idx] = 1.0
        diag_offset = col_idx * n + col_idx
        tl.store(aa_ptr + diag_offset, 1.0)

def s2102_triton(aa):
    n = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s2102_kernel[grid](aa, n, BLOCK_SIZE)
    
    return aa
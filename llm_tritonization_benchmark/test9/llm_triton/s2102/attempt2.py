import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Process columns assigned to this block
    for col_idx in range(block_start, min(block_start + BLOCK_SIZE, n)):
        if col_idx < n:
            # Zero out entire column col_idx
            for row_block in range(0, n, BLOCK_SIZE):
                row_offsets = row_block + tl.arange(0, BLOCK_SIZE)
                row_mask = row_offsets < n
                
                # aa[row_offsets][col_idx] = 0.0
                store_offsets = row_offsets * n + col_idx
                tl.store(aa_ptr + store_offsets, 0.0, mask=row_mask)
            
            # Set diagonal element aa[col_idx][col_idx] = 1.0
            diag_offset = col_idx * n + col_idx
            tl.store(aa_ptr + diag_offset, 1.0)

def s2102_triton(aa):
    n = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s2102_kernel[grid](aa, n, BLOCK_SIZE)
    
    return aa
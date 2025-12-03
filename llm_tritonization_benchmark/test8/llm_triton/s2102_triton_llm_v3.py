import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get block position and offsets
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks of columns
    col_start = block_id * BLOCK_SIZE
    cols = col_start + offsets
    col_mask = cols < LEN_2D
    
    # Process each column in this block
    for col_offset in range(BLOCK_SIZE):
        col = col_start + col_offset
        if col < LEN_2D:
            # Zero out the entire column
            for row_start in range(0, LEN_2D, BLOCK_SIZE):
                row_offsets = row_start + offsets
                row_mask = row_offsets < LEN_2D
                
                # Calculate flat indices for column-major access: aa[j][i] = aa[j * LEN_2D + i]
                flat_indices = row_offsets * LEN_2D + col
                
                # Store zeros
                tl.store(aa_ptr + flat_indices, 0.0, mask=row_mask)
            
            # Set diagonal element to 1.0: aa[i][i] = 1.0
            diagonal_idx = col * LEN_2D + col
            tl.store(aa_ptr + diagonal_idx, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Launch kernel with one block per set of columns
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2102_kernel[grid](aa, LEN_2D, BLOCK_SIZE)
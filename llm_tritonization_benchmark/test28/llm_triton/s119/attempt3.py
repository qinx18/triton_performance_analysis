import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, len_2d, diag, start_i, BLOCK_SIZE: tl.constexpr):
    # Get position within this diagonal
    pos_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual i coordinate
    i = start_i + pos_idx
    
    # Calculate j coordinate
    j = diag - i
    
    # Bounds check
    mask = (i < len_2d) & (j < len_2d) & (i >= 1) & (j >= 1) & (pos_idx < (len_2d - start_i))
    
    # Load values
    aa_prev = tl.load(aa_ptr + (i - 1) * len_2d + (j - 1), mask=mask)
    bb_val = tl.load(bb_ptr + i * len_2d + j, mask=mask)
    
    # Compute and store result
    result = aa_prev + bb_val
    tl.store(aa_ptr + i * len_2d + j, result, mask=mask)

def s119_triton(aa, bb, len_2d):
    BLOCK_SIZE = 256
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * len_2d):  # diag = i + j
        # Calculate range of valid i values for this diagonal
        start_i = max(1, diag - len_2d + 1)
        end_i = min(diag, len_2d)
        
        # Number of elements in this diagonal
        num_elements = end_i - start_i
        
        if num_elements <= 0:
            continue
            
        # Launch kernel for this diagonal
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        s119_kernel[grid](
            aa, bb, len_2d, diag, start_i,
            BLOCK_SIZE=BLOCK_SIZE
        )
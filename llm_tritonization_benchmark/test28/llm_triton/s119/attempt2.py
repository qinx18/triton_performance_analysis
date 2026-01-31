import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get the current anti-diagonal and position within it
    diag_idx = tl.program_id(0)
    pos_idx = tl.program_id(1)
    
    # Calculate actual diagonal value (starts from 2 since i,j start from 1)
    diag = diag_idx + 2
    
    # Calculate the range of valid i values for this diagonal
    start_i = tl.maximum(1, diag - len_2d + 1)
    end_i = tl.minimum(diag, len_2d)
    
    # Calculate actual i coordinate
    i = start_i + pos_idx
    
    # Check if this position is valid
    if i >= end_i:
        return
    
    # Calculate j coordinate
    j = diag - i
    
    # Bounds check
    if (i >= len_2d) | (j >= len_2d) | (i < 1) | (j < 1):
        return
    
    # Load values
    aa_prev = tl.load(aa_ptr + (i - 1) * len_2d + (j - 1))
    bb_val = tl.load(bb_ptr + i * len_2d + j)
    
    # Compute and store result
    result = aa_prev + bb_val
    tl.store(aa_ptr + i * len_2d + j, result)

def s119_triton(aa, bb, len_2d):
    # Process using anti-diagonal parallelization
    # Each anti-diagonal k = i + j can be processed in parallel
    # since elements don't depend on each other within the same diagonal
    
    BLOCK_SIZE = 256
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * len_2d - 1):  # diag = i + j, starting from 2
        # Calculate range of valid i values for this diagonal
        start_i = max(1, diag - len_2d + 1)
        end_i = min(diag, len_2d)
        
        # Number of elements in this diagonal
        num_elements = end_i - start_i
        
        if num_elements <= 0:
            continue
            
        # Launch kernel for this diagonal
        grid = (1, triton.cdiv(num_elements, BLOCK_SIZE))
        s119_kernel[grid](
            aa, bb, len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )
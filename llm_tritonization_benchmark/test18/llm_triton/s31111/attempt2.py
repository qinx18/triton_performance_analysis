import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single block computes the entire sum
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Load all 32 elements at once (8 groups of 4)
    offsets = tl.arange(0, 32)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Sum each group of 4 elements
    # Group 1: elements 0-3
    sum_val += vals[0] + vals[1] + vals[2] + vals[3]
    
    # Group 2: elements 4-7  
    sum_val += vals[4] + vals[5] + vals[6] + vals[7]
    
    # Group 3: elements 8-11
    sum_val += vals[8] + vals[9] + vals[10] + vals[11]
    
    # Group 4: elements 12-15
    sum_val += vals[12] + vals[13] + vals[14] + vals[15]
    
    # Group 5: elements 16-19
    sum_val += vals[16] + vals[17] + vals[18] + vals[19]
    
    # Group 6: elements 20-23
    sum_val += vals[20] + vals[21] + vals[22] + vals[23]
    
    # Group 7: elements 24-27
    sum_val += vals[24] + vals[25] + vals[26] + vals[27]
    
    # Group 8: elements 28-31
    sum_val += vals[28] + vals[29] + vals[30] + vals[31]
    
    # Store result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 32
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch single block
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
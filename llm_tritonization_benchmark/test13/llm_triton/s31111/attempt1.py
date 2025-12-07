import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    sum_val = 0.0
    
    # Process each group of 4 elements starting at positions 0, 4, 8, 12, 16, 20, 24, 28
    for start_idx in range(8):
        base_idx = start_idx * 4
        
        # Load 4 consecutive elements for this test() call
        test_offsets = block_start + base_idx + offsets
        mask = (test_offsets < n_elements) & (offsets < 4)
        
        vals = tl.load(a_ptr + test_offsets, mask=mask, other=0.0)
        
        # Sum the 4 elements (equivalent to test() function)
        test_sum = tl.sum(vals)
        sum_val += test_sum
    
    # Store result back to first element of each block
    if offsets[0] == 0:
        result_offset = block_start
        if result_offset < n_elements:
            tl.store(a_ptr + result_offset, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 32
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s31111_kernel[grid](
        a, n_elements, BLOCK_SIZE
    )
    
    return a
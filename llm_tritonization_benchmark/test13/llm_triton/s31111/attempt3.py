import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = 0.0
    
    # Process each group of 4 elements starting at positions 0, 4, 8, 12, 16, 20, 24, 28
    for start_idx in range(8):
        base_idx = start_idx * 4
        
        # Load 4 consecutive elements for this test() call
        test_offsets = base_idx + tl.arange(0, 4)
        mask = test_offsets < n_elements
        
        vals = tl.load(a_ptr + test_offsets, mask=mask, other=0.0)
        
        # Sum the 4 elements (equivalent to test() function)
        test_sum = tl.sum(vals)
        sum_val += test_sum
    
    # Store result
    if pid == 0:
        tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 32
    
    # Launch kernel with single block
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    # Store result back to a[0]
    a[0] = result[0]
    
    return a
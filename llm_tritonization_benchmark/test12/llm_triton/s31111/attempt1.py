import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel computes sum of elements at specific offsets
    # sum += test(a) + test(&a[4]) + test(&a[8]) + ... + test(&a[28])
    # where test(ptr) sums 4 consecutive elements starting at ptr
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Process each group of 4 elements starting at offsets 0, 4, 8, 12, 16, 20, 24, 28
    for start_offset in range(0, 32, 4):
        group_sum = 0.0
        # Sum 4 consecutive elements starting at start_offset
        for i in range(4):
            offset = start_offset + i
            if offset < n_elements:
                val = tl.load(a_ptr + offset)
                group_sum += val
        sum_val += group_sum
    
    # Store the result
    tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    
    # Create output tensor for sum
    sum_result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel with single block
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, n_elements, BLOCK_SIZE
    )
    
    return sum_result[0].item()
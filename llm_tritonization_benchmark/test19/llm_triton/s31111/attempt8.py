import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    # Calculate sum of elements at specific offsets
    # Each test call sums 4 consecutive elements starting at different bases
    
    sum_val = 0.0
    
    # Define offsets for loading 4 consecutive elements
    offsets = tl.arange(0, 4)
    mask = offsets < 4
    
    # sum += test(a) - sum first 4 elements starting at index 0
    vals = tl.load(a_ptr + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    # sum += test(&a[4]) - sum 4 elements starting at index 4
    vals = tl.load(a_ptr + 4 + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    # sum += test(&a[8]) - sum 4 elements starting at index 8
    vals = tl.load(a_ptr + 8 + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    # sum += test(&a[12]) - sum 4 elements starting at index 12
    vals = tl.load(a_ptr + 12 + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    # sum += test(&a[16]) - sum 4 elements starting at index 16
    vals = tl.load(a_ptr + 16 + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    # sum += test(&a[20]) - sum 4 elements starting at index 20
    vals = tl.load(a_ptr + 20 + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    # sum += test(&a[24]) - sum 4 elements starting at index 24
    vals = tl.load(a_ptr + 24 + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    # sum += test(&a[28]) - sum 4 elements starting at index 28
    vals = tl.load(a_ptr + 28 + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    # Store the final sum
    tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single thread since we're computing one sum
    grid = (1,)
    s31111_kernel[grid](a, sum_result, BLOCK_SIZE=32)
    
    return sum_result.item()
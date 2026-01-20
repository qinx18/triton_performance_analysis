import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    # Calculate sum of elements at specific offsets
    # Each offset group: a[i], a[i+4], a[i+8], ..., a[i+28] where i is the base offset
    
    sum_val = 0.0
    
    # For each base offset, sum 4 consecutive elements (test function)
    base_start = 0
    offsets = tl.arange(0, 4)
    
    mask = offsets < 4
    vals = tl.load(a_ptr + base_start + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    base_start = 4
    vals = tl.load(a_ptr + base_start + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    base_start = 8
    vals = tl.load(a_ptr + base_start + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    base_start = 12
    vals = tl.load(a_ptr + base_start + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    base_start = 16
    vals = tl.load(a_ptr + base_start + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    base_start = 20
    vals = tl.load(a_ptr + base_start + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    base_start = 24
    vals = tl.load(a_ptr + base_start + offsets, mask=mask)
    sum_val += tl.sum(vals)
    
    base_start = 28
    vals = tl.load(a_ptr + base_start + offsets, mask=mask)
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
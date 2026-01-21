import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, BLOCK_SIZE: tl.constexpr):
    # Initialize sum
    sum_val = 0.0
    
    # Create offsets for loading 4 elements at a time
    offsets = tl.arange(0, 4)
    
    # Sum elements 0,1,2,3
    vals = tl.load(a_ptr + offsets)
    sum_val += tl.sum(vals)
    
    # Sum elements 4,5,6,7
    vals = tl.load(a_ptr + offsets + 4)
    sum_val += tl.sum(vals)
    
    # Sum elements 8,9,10,11
    vals = tl.load(a_ptr + offsets + 8)
    sum_val += tl.sum(vals)
    
    # Sum elements 12,13,14,15
    vals = tl.load(a_ptr + offsets + 12)
    sum_val += tl.sum(vals)
    
    # Sum elements 16,17,18,19
    vals = tl.load(a_ptr + offsets + 16)
    sum_val += tl.sum(vals)
    
    # Sum elements 20,21,22,23
    vals = tl.load(a_ptr + offsets + 20)
    sum_val += tl.sum(vals)
    
    # Sum elements 24,25,26,27
    vals = tl.load(a_ptr + offsets + 24)
    sum_val += tl.sum(vals)
    
    # Sum elements 28,29,30,31
    vals = tl.load(a_ptr + offsets + 28)
    sum_val += tl.sum(vals)
    
    # Store result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result[0].item()
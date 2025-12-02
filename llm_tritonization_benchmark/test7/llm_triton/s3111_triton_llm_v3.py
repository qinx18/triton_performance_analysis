import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition: only add if val > 0
        condition = vals > 0.0
        conditional_vals = tl.where(condition, vals, 0.0)
        
        sum_val += tl.sum(conditional_vals, axis=0)
    
    tl.store(result_ptr, sum_val)

def s3111_triton(a):
    n_elements = a.numel()
    
    # Create output tensor for the sum
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s3111_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
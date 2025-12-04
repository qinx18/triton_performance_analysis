import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel - sum all elements of array a
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load elements from array a
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Sum within this block
    block_sum = tl.sum(a_vals)
    
    # Store the partial sum
    if tl.program_id(0) == 0:
        tl.atomic_add(result_ptr, block_sum)
    else:
        tl.atomic_add(result_ptr, block_sum)

def s311_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s311_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()
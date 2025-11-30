import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, max_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values and compute absolute values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(a_vals)
    
    # Reduce to find maximum within block
    block_max = tl.max(abs_vals, axis=0)
    
    # Store block maximum
    tl.atomic_max(max_ptr, block_max)

def s3113_triton(a):
    n_elements = a.numel()
    
    # Initialize max with absolute value of first element
    max_result = torch.abs(a[0:1]).clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3113_kernel[grid](
        a, max_result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return max_result.item()
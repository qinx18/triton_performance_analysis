import triton
import triton.language as tl
import torch

@triton.jit
def s151_kernel(
    a_ptr, b_ptr,
    n_elements,
    inc,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_val = tl.load(a_ptr + offsets, mask=mask)
    b_val = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform computation: a[i] = a[i] + b[i] * inc
    result = a_val + b_val * inc
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.numel()
    inc = 1.0
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s151_kernel[grid](
        a, b,
        n_elements,
        inc,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a
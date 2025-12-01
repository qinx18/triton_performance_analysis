import triton
import triton.language as tl
import torch

@triton.jit
def s151_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    # Perform the computation: a[i] = a[i] + b[i] * inc
    result = a_vals + b_vals * inc
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.numel()
    inc = 1.0
    
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s151_kernel[grid](
        a, b, inc, n_elements, BLOCK_SIZE
    )
    
    return a
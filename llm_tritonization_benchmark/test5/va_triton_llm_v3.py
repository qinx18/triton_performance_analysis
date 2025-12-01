import triton
import triton.language as tl
import torch

@triton.jit
def va_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    tl.store(a_ptr + indices, b_vals, mask=mask)

def va_triton(a, b):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    va_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
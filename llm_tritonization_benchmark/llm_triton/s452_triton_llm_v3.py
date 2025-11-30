import triton
import triton.language as tl
import torch

@triton.jit
def s452_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute (i+1) for each element
    indices = offsets + 1
    
    # Compute a[i] = b[i] + c[i] * (i+1)
    result = b_vals + c_vals * indices.to(tl.float32)
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s452_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s452_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
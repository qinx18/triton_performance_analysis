import torch
import triton
import triton.language as tl

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load a[j] where j = i + 1, so a[i+1]
    a_vals = tl.load(a_ptr + offsets + 1, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
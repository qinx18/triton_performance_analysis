import triton
import triton.language as tl
import torch

@triton.jit
def s173_kernel(a_ptr, b_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[i] and b[i]
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+k]
    tl.store(a_ptr + offsets + k, result, mask=mask)

def s173_triton(a, b):
    n_elements = a.shape[0] // 2
    k = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s173_kernel[grid](
        a, b, n_elements, k, BLOCK_SIZE=BLOCK_SIZE
    )
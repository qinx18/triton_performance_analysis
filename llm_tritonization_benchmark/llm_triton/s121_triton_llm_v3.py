import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[i+1] (j = i + 1)
    a_offsets = offsets + 1
    a_mask = a_offsets < (n_elements + 1)
    a_vals = tl.load(a_ptr + a_offsets, mask=a_mask)
    
    # Load b[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[j] + b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a
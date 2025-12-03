import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load a[i+1] values (a[j] where j = i + 1)
    a_next_indices = indices + 1
    a_next_mask = a_next_indices < (n_elements + 1)
    a_next_vals = tl.load(a_ptr + a_next_indices, mask=a_next_mask)
    
    # Compute a[i] = a[j] + b[i] where j = i + 1
    result = a_next_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a
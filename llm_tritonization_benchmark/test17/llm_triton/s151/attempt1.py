import torch
import triton
import triton.language as tl

@triton.jit
def s151_kernel(
    a_ptr, b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices (process n_elements-1 elements)
    mask = indices < (n_elements - 1)
    
    # Load a[i+1] and b[i]
    a_vals = tl.load(a_ptr + indices + 1, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.shape[0]
    
    # We process n_elements-1 elements
    elements_to_process = n_elements - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s151_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
import triton
import triton.language as tl
import torch

@triton.jit
def s431_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Current indices
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Read from a[i+k] and b[i]
    a_indices = indices + k
    a_mask = (indices < n_elements) & (a_indices < n_elements + k)
    
    a_vals = tl.load(a_ptr + a_indices, mask=a_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[i+k] + b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s431_triton(a, b):
    n_elements = a.shape[0]
    k = 0  # k = 2*k1 - k2 = 2*1 - 2 = 0
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a,
        b,
        n_elements,
        k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
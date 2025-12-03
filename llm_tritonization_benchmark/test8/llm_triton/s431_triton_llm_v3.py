import torch
import triton
import triton.language as tl

@triton.jit
def s431_kernel(a_ptr, b_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load a[i+k] and b[i]
    a_offsets = block_start + offsets + k
    a_mask = mask & (a_offsets < n_elements + k)
    a_vals = tl.load(a_ptr + a_offsets, mask=a_mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    # Compute a[i] = a[i+k] + b[i]
    result = a_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s431_triton(a, b, k):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, b, n_elements, k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
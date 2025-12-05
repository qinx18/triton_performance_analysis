import triton
import triton.language as tl
import torch

@triton.jit
def s431_kernel(a_ptr, b_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load b[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Load a[i+k]
    read_idx = idx + k
    read_mask = mask & (read_idx < n_elements + k)
    a_read_vals = tl.load(a_ptr + read_idx, mask=read_mask)
    
    # Compute a[i] = a[i+k] + b[i]
    result = a_read_vals + b_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s431_triton(a, b, k):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, b, n_elements, k,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
import torch
import triton
import triton.language as tl

@triton.jit
def s431_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load b[i]
    b_ptrs = b_ptr + block_start + offsets
    b_vals = tl.load(b_ptrs, mask=mask)
    
    # Load a[i+k] where k=0
    a_read_ptrs = a_ptr + block_start + offsets
    a_vals = tl.load(a_read_ptrs, mask=mask)
    
    # Compute a[i] = a[i+k] + b[i]
    result = a_vals + b_vals
    
    # Store result to a[i]
    a_store_ptrs = a_ptr + block_start + offsets
    tl.store(a_store_ptrs, result, mask=mask)

def s431_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a
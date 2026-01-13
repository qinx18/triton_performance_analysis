import triton
import triton.language as tl
import torch

@triton.jit
def s431_kernel(a_ptr, a_copy_ptr, b_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load from read-only copy at offset positions
    a_vals = tl.load(a_copy_ptr + idx + k, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s431_triton(a, b, k):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, a_copy, b, k, n_elements, BLOCK_SIZE
    )
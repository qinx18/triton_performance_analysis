import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from read-only copy for all reads
    a_vals = tl.load(a_copy_ptr + offsets, mask=mask)
    a_vals_next = tl.load(a_copy_ptr + offsets + 1, mask=mask)
    
    # Compute: a[i] = a[i + 1] * a[i]
    result = a_vals_next * a_vals
    
    # Store to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s116_triton(a):
    n_elements = a.shape[0] - 5
    
    # Create read-only copy to avoid WAR race conditions
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s116_kernel[grid](
        a, a_copy, n_elements, BLOCK_SIZE
    )
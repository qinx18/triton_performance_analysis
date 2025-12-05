import torch
import triton
import triton.language as tl

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[0] from the copy (read-only)
    a_zero = tl.load(a_copy_ptr)
    
    # Store a[0] to all elements a[i]
    tl.store(a_ptr + offsets, a_zero, mask=mask)

def s293_triton(a):
    n_elements = a.numel()
    
    # Create read-only copy to avoid WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a, a_copy, n_elements, BLOCK_SIZE
    )
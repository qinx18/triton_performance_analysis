import torch
import triton
import triton.language as tl

@triton.jit
def s293_kernel(
    a_ptr,
    a_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load a[0] from the read-only copy
    a0_val = tl.load(a_copy_ptr)
    
    # Store a[0] value to all positions a[i]
    tl.store(a_ptr + indices, a0_val, mask=mask)

def s293_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to avoid WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
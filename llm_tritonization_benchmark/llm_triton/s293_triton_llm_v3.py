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
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[0] from the copy
    a0_val = tl.load(a_copy_ptr)
    
    # Store a[0] to all positions in this block
    tl.store(a_ptr + offsets, a0_val, mask=mask)

def s293_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
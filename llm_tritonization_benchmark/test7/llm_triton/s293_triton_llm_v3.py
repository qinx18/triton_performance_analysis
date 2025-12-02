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
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Read a[0] from the copy
        a0_val = tl.load(a_copy_ptr)
        
        # Store a[0] to all positions a[i]
        tl.store(a_ptr + current_offsets, a0_val, mask=mask)

def s293_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
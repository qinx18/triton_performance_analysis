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
    block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load a[0] from the copy (all threads load the same value)
    a_0_val = tl.load(a_copy_ptr)
    
    # Store a_0_val to all positions a[i]
    tl.store(a_ptr + idx, a_0_val, mask=mask)

def s293_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
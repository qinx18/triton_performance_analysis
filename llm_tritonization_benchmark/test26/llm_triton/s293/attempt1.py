import triton
import triton.language as tl
import torch

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load the original a[0] value from the copy
    a0_value = tl.load(a_copy_ptr)
    
    # Store a[0] to all elements a[i]
    tl.store(a_ptr + indices, a0_value, mask=mask)

def s293_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a, a_copy, n_elements, BLOCK_SIZE
    )
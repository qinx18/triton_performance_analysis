import torch
import triton
import triton.language as tl

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate block start
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid elements
    mask = indices < n_elements
    
    # Load a[0] from the read-only copy
    a0_value = tl.load(a_copy_ptr)
    
    # Broadcast a[0] to all elements in the block
    values = tl.full([BLOCK_SIZE], a0_value, dtype=tl.float32)
    
    # Store to output array
    tl.store(a_ptr + indices, values, mask=mask)

def s293_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s293_kernel[grid](
        a, a_copy, n_elements, BLOCK_SIZE
    )
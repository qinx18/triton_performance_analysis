import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(
    a_ptr,
    a_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE * 5
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 5
    mask = offsets < (n_elements - 5)
    
    # Load original values from read-only copy
    a_i = tl.load(a_copy_ptr + offsets, mask=mask)
    a_i1 = tl.load(a_copy_ptr + offsets + 1, mask=mask)
    a_i2 = tl.load(a_copy_ptr + offsets + 2, mask=mask)
    a_i3 = tl.load(a_copy_ptr + offsets + 3, mask=mask)
    a_i4 = tl.load(a_copy_ptr + offsets + 4, mask=mask)
    a_i5 = tl.load(a_copy_ptr + offsets + 5, mask=mask)
    
    # Compute updates according to dependencies
    new_a_i = a_i1 * a_i
    new_a_i1 = a_i2 * a_i1
    new_a_i2 = a_i3 * a_i2
    new_a_i3 = a_i4 * a_i3
    new_a_i4 = a_i5 * a_i4
    
    # Store results to original array
    tl.store(a_ptr + offsets, new_a_i, mask=mask)
    tl.store(a_ptr + offsets + 1, new_a_i1, mask=mask)
    tl.store(a_ptr + offsets + 2, new_a_i2, mask=mask)
    tl.store(a_ptr + offsets + 3, new_a_i3, mask=mask)
    tl.store(a_ptr + offsets + 4, new_a_i4, mask=mask)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    num_blocks = (n_elements + BLOCK_SIZE * 5 - 1) // (BLOCK_SIZE * 5)
    
    grid = (num_blocks,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE,
    )
    
    return a
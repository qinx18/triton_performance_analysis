import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE + 1
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Read from copy for a[i-1] and original arrays for b[i]
    a_prev_offsets = offsets - 1
    a_prev_vals = tl.load(a_copy_ptr + a_prev_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_vals = tl.load(a_copy_ptr + offsets, mask=mask)
    
    # Compute: a[i] += a[i-1] * b[i]
    result = a_vals + a_prev_vals * b_vals
    
    # Store to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s321_triton(a, b):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = ((n_elements - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s321_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE,
    )
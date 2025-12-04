import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    m,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Read from copy and b arrays
    a_copy_vals = tl.load(a_copy_ptr + current_offsets + m, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # Compute result
    result = a_copy_vals + b_vals
    
    # Store to original a array
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s131_triton(a, b, m):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a,
        a_copy,
        b,
        m,
        n_elements,
        BLOCK_SIZE,
    )
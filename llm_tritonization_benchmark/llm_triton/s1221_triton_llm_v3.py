import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(
    b_ptr,
    b_copy_ptr,
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE + 4
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Read from copy for b[i-4] and from original for a[i]
    b_prev_offsets = offsets - 4
    b_prev_vals = tl.load(b_copy_ptr + b_prev_offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute result
    result = b_prev_vals + a_vals
    
    # Store to original b array
    tl.store(b_ptr + offsets, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements - 4, meta['BLOCK_SIZE']),)
    
    s1221_kernel[grid](
        b, b_copy, a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
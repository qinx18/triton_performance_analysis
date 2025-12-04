import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(
    a_ptr, b_ptr, c_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (must be < n_elements - 1)
    mask = offsets < (n_elements - 1)
    
    # Load b[i], c[i], e[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute and store a[i+1] = b[i] + e[i] FIRST
    a_i_plus_1_vals = b_vals + e_vals
    a_i_plus_1_offsets = offsets + 1
    tl.store(a_ptr + a_i_plus_1_offsets, a_i_plus_1_vals, mask=mask)
    
    # Then compute and store a[i] = b[i] + c[i]
    a_i_vals = b_vals + c_vals
    tl.store(a_ptr + offsets, a_i_vals, mask=mask)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size based on n_elements - 1
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    # Launch kernel
    s2244_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a
import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the range of elements this program will process
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute the conditional expression
    condition = b_vals > 0.0
    de_product = d_vals * e_vals
    result = tl.where(condition, b_vals + de_product, c_vals + de_product)
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
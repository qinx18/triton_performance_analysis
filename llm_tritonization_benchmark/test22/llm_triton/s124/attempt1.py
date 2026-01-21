import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program processes one block of elements
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + indices, mask=mask, other=0.0)
    
    # Compute the multiplication
    de_product = d_vals * e_vals
    
    # Apply conditional selection
    condition = b_vals > 0.0
    result = tl.where(condition, b_vals + de_product, c_vals + de_product)
    
    # Store results (output indices are same as input indices)
    tl.store(a_ptr + indices, result, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
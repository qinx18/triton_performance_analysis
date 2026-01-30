import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(b_ptr, c_ptr, d_ptr, e_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute cumulative sum for each element to get output indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + indices, mask=mask, other=0.0)
    
    # Compute values
    de_vals = d_vals * e_vals
    result_vals = tl.where(b_vals > 0.0, b_vals + de_vals, c_vals + de_vals)
    
    # Store results (output index equals input index since no filtering)
    tl.store(a_ptr + indices, result_vals, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        b, c, d, e, a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
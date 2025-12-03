import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    i_mask = i_offsets < n_elements
    
    # Load input values
    b_vals = tl.load(b_ptr + i_offsets, mask=i_mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=i_mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=i_mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Calculate j indices: j starts at 0 for first element, then increments by 2
    j_even_offsets = 2 * i_offsets
    j_odd_offsets = 2 * i_offsets + 1
    
    # Calculate output values
    a_even_vals = b_vals + c_vals * d_vals
    a_odd_vals = b_vals + d_vals * e_vals
    
    # Store to even j indices (j = 0, 2, 4, ...)
    j_even_mask = i_mask & (j_even_offsets < (2 * n_elements))
    tl.store(a_ptr + j_even_offsets, a_even_vals, mask=j_even_mask)
    
    # Store to odd j indices (j = 1, 3, 5, ...)
    j_odd_mask = i_mask & (j_odd_offsets < (2 * n_elements))
    tl.store(a_ptr + j_odd_offsets, a_odd_vals, mask=j_odd_mask)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
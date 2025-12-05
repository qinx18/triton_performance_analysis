import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Load values for index i
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Compute j indices: j = 2*i and j = 2*i + 1
    j_even = 2 * i_offsets
    j_odd = 2 * i_offsets + 1
    
    # Compute values
    a_even = b_vals + c_vals * d_vals
    a_odd = b_vals + d_vals * e_vals
    
    # Store to even j positions
    even_mask = mask & (j_even < 2 * n_elements)
    tl.store(a_ptr + j_even, a_even, mask=even_mask)
    
    # Store to odd j positions
    odd_mask = mask & (j_odd < 2 * n_elements)
    tl.store(a_ptr + j_odd, a_odd, mask=odd_mask)

def s127_triton(a, b, c, d, e):
    n_elements = b.numel() // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
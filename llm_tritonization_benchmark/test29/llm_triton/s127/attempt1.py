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
    
    # Load data for iteration i
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Compute j = 2*i and j = 2*i + 1 offsets
    j_even_offsets = 2 * i_offsets
    j_odd_offsets = 2 * i_offsets + 1
    
    # Compute values
    val_even = b_vals + c_vals * d_vals
    val_odd = b_vals + d_vals * e_vals
    
    # Store at even positions (j = 2*i)
    tl.store(a_ptr + j_even_offsets, val_even, mask=mask)
    
    # Store at odd positions (j = 2*i + 1)
    tl.store(a_ptr + j_odd_offsets, val_odd, mask=mask)

def s127_triton(a, b, c, d, e):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
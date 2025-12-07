import torch
import triton
import triton.language as tl

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Load values for this block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Calculate j indices: j = 2*i, j+1 = 2*i+1
    j_even_offsets = 2 * i_offsets
    j_odd_offsets = 2 * i_offsets + 1
    
    # Calculate values
    a_even_vals = b_vals + c_vals * d_vals
    a_odd_vals = b_vals + d_vals * e_vals
    
    # Store to a[j] = a[2*i] and a[j+1] = a[2*i+1]
    tl.store(a_ptr + j_even_offsets, a_even_vals, mask=mask)
    tl.store(a_ptr + j_odd_offsets, a_odd_vals, mask=mask)

def s127_triton(a, b, c, d, e):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
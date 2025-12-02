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
    
    # Load data for current block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Compute j indices: j = 2*i for first equation, j = 2*i+1 for second equation
    j_even = 2 * i_offsets  # j for first equation
    j_odd = 2 * i_offsets + 1  # j for second equation
    
    # Compute values
    val1 = b_vals + c_vals * d_vals  # a[j] = b[i] + c[i] * d[i]
    val2 = b_vals + d_vals * e_vals  # a[j] = b[i] + d[i] * e[i]
    
    # Store to even j positions (2*i)
    even_mask = mask & (j_even < 2 * n_elements)
    tl.store(a_ptr + j_even, val1, mask=even_mask)
    
    # Store to odd j positions (2*i+1)  
    odd_mask = mask & (j_odd < 2 * n_elements)
    tl.store(a_ptr + j_odd, val2, mask=odd_mask)

def s127_triton(a, b, c, d, e):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
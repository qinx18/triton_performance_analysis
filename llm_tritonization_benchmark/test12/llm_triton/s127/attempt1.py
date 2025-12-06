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
    
    # Load values for index i
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Calculate j indices: j starts at -1, then increments by 1 for each operation
    # For iteration i: first j = 2*i, second j = 2*i + 1
    j1_offsets = 2 * i_offsets
    j2_offsets = 2 * i_offsets + 1
    
    # Calculate values
    val1 = b_vals + c_vals * d_vals
    val2 = b_vals + d_vals * e_vals
    
    # Store results
    tl.store(a_ptr + j1_offsets, val1, mask=mask)
    tl.store(a_ptr + j2_offsets, val2, mask=mask)

def s127_triton(a, b, c, d, e):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
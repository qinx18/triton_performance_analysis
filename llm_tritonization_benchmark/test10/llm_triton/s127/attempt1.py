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
    
    # Load input values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute j indices: j starts at -1, then increments by 1 for each assignment
    # For i-th iteration: first j = 2*i, second j = 2*i + 1
    j_offsets_1 = 2 * i_offsets
    j_offsets_2 = 2 * i_offsets + 1
    
    # Compute values
    val1 = b_vals + c_vals * d_vals
    val2 = b_vals + d_vals * e_vals
    
    # Store results
    mask_j1 = j_offsets_1 < (2 * n_elements)
    mask_j2 = j_offsets_2 < (2 * n_elements)
    
    tl.store(a_ptr + j_offsets_1, val1, mask=mask & mask_j1)
    tl.store(a_ptr + j_offsets_2, val2, mask=mask & mask_j2)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
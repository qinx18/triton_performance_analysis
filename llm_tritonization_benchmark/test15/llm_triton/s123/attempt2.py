import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid elements
    mask = i_offsets < n_elements
    
    # Load input values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Calculate j offsets for first store: j starts at block_start * 2
    j_base = block_start * 2
    j_offsets_1 = j_base + offsets * 2
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    result1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets_1, result1, mask=mask)
    
    # Second assignment for positive c values
    c_positive = c_vals > 0.0
    combined_mask = mask & c_positive
    j_offsets_2 = j_offsets_1 + 1
    result2 = c_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets_2, result2, mask=combined_mask)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # For each i, first assignment is always at a[2*i] (j starts at -1, then j++ makes it 0 for i=0)
    j_offsets_1 = 2 * offsets
    first_vals = b_vals + d_vals * e_vals
    first_mask = (j_offsets_1 < 2 * n_elements) & mask
    tl.store(a_ptr + j_offsets_1, first_vals, mask=first_mask)
    
    # Second assignment happens at a[2*i + 1] when c[i] > 0
    j_offsets_2 = 2 * offsets + 1
    second_vals = c_vals + d_vals * e_vals
    second_mask = (j_offsets_2 < 2 * n_elements) & mask & (c_vals > 0.0)
    tl.store(a_ptr + j_offsets_2, second_vals, mask=second_mask)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
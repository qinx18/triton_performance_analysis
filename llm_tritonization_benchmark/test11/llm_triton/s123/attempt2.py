import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Calculate j offsets for first assignment
    j_offsets = i_offsets * 2
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    val1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets, val1, mask=mask)
    
    # Second assignment when c[i] > 0
    cond_mask = mask & (c_vals > 0.0)
    j_offsets_2 = j_offsets + 1
    val2 = c_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets_2, val2, mask=cond_mask)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
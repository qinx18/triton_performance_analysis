import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offset = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = i_offset + offsets
    mask = i_offsets < n_elements
    
    # Load input arrays
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Calculate j values for each element
    # j starts at -1, then incremented for each i
    j_base = i_offsets * 2
    
    # First store: a[j] = b[i] + d[i] * e[i]
    result1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_base, result1, mask=mask)
    
    # Second store for c[i] > 0: a[j+1] = c[i] + d[i] * e[i]
    c_positive = c_vals > 0.0
    result2 = c_vals + d_vals * e_vals
    store_mask = mask & c_positive
    tl.store(a_ptr + j_base + 1, result2, mask=store_mask)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
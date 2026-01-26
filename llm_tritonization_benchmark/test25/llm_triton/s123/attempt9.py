import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < n_elements
    
    # Load input arrays
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute j = 2*i (since j starts at -1 and increments before first use)
    j_first = 2 * i_offsets
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    first_vals = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_first, first_vals, mask=mask)
    
    # Conditional assignment: if c[i] > 0, a[j+1] = c[i] + d[i] * e[i]
    cond_mask = mask & (c_vals > 0.0)
    j_second = j_first + 1
    second_vals = c_vals + d_vals * e_vals
    tl.store(a_ptr + j_second, second_vals, mask=cond_mask)

def s123_triton(a, b, c, d, e):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 256
    
    # Clear array a
    a.zero_()
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load values for this block
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate j indices: j starts at -1, then increments for each i
    # For element i, j = i (after first increment)
    j_indices = offsets
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    first_vals = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_indices, first_vals, mask=mask)
    
    # Conditional assignment: if c[i] > 0, then j++ and store again
    cond_mask = mask & (c_vals > 0.0)
    j_indices_cond = offsets + n_elements  # Offset to avoid conflicts
    second_vals = c_vals + d_vals * e_vals
    tl.store(a_ptr + j_indices_cond, second_vals, mask=cond_mask)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # LEN_1D/2
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
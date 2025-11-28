import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Load b[i+1] for the second computation
    offsets_plus1 = offsets + 1
    mask_plus1 = offsets_plus1 < (n_elements + 1)
    
    # Control flow logic
    # if (a[i] >= 0.) goto L20 - skip both computations
    skip_all = a_vals >= 0.0
    
    # if (b[i] >= 0.) goto L30 - skip first computation but do second
    skip_first_only = (a_vals < 0.0) & (b_vals >= 0.0)
    
    # Do both computations when both conditions are false
    do_both = (a_vals < 0.0) & (b_vals < 0.0)
    
    # First computation: a[i] += c[i] * d[i] (only when do_both is true)
    new_a_vals = tl.where(do_both, a_vals + c_vals * d_vals, a_vals)
    
    # Second computation: b[i+1] = c[i] + d[i] * e[i] (when skip_first_only OR do_both)
    compute_second = skip_first_only | do_both
    b_new_vals = c_vals + d_vals * e_vals
    
    # Store updated a values
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    
    # Store updated b[i+1] values where needed
    tl.store(b_ptr + offsets_plus1, b_new_vals, mask=mask_plus1 & compute_second)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
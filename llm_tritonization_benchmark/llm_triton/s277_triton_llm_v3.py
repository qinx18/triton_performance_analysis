import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
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
    
    # Load b[i+1] for the update (need to check bounds)
    offsets_plus_one = offsets + 1
    mask_plus_one = offsets_plus_one < (n_elements + 1)
    b_plus_one = tl.load(b_ptr + offsets_plus_one, mask=mask_plus_one)
    
    # Implement the control flow logic
    # if (a[i] >= 0.0) goto L20 (skip everything)
    a_skip = a_vals >= 0.0
    
    # if (b[i] >= 0.0) goto L30 (skip a[i] update, but do b[i+1] update)
    b_skip = b_vals >= 0.0
    
    # a[i] += c[i] * d[i] (only if both conditions are false)
    should_update_a = ~a_skip & ~b_skip
    a_new = tl.where(should_update_a, a_vals + c_vals * d_vals, a_vals)
    
    # L30: b[i+1] = c[i] + d[i] * e[i] (only if a[i] < 0.0)
    should_update_b = ~a_skip
    b_plus_one_new = tl.where(should_update_b, c_vals + d_vals * e_vals, b_plus_one)
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets_plus_one, b_plus_one_new, mask=mask_plus_one)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b
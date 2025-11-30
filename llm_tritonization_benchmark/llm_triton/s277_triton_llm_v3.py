import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Load b[i+1] for the second operation
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)
    
    # Control flow logic
    # if (a[i] >= 0.0) goto L20 (skip everything)
    a_ge_zero = a_vals >= 0.0
    
    # if (b[i] >= 0.0) goto L30 (skip a[i] update)
    b_ge_zero = b_vals >= 0.0
    
    # a[i] += c[i] * d[i] (only if a[i] < 0 and b[i] < 0)
    update_a_mask = (~a_ge_zero) & (~b_ge_zero)
    new_a_vals = tl.where(update_a_mask, a_vals + c_vals * d_vals, a_vals)
    
    # L30: b[i+1] = c[i] + d[i] * e[i] (only if a[i] < 0)
    update_b_mask = ~a_ge_zero
    new_b_vals = c_vals + d_vals * e_vals
    
    # Store updated a values
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    
    # Store updated b[i+1] values
    tl.store(b_ptr + offsets_plus_1, new_b_vals, mask=mask & update_b_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D-1 from the loop
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
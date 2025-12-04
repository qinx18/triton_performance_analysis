import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Load b[i+1] values for the update
    b_next_idx = idx + 1
    b_next_mask = mask & (b_next_idx < n_elements + 1)
    
    # Implement the control flow logic
    # if (a[i] >= 0.) goto L20 - skip both operations
    a_nonneg = a_vals >= 0.0
    
    # if (b[i] >= 0.) goto L30 - skip first operation but do second
    b_nonneg = b_vals >= 0.0
    
    # First operation: a[i] += c[i] * d[i] (only if a[i] < 0 AND b[i] < 0)
    do_first_op = (~a_nonneg) & (~b_nonneg) & mask
    new_a_vals = tl.where(do_first_op, a_vals + c_vals * d_vals, a_vals)
    
    # Second operation: b[i+1] = c[i] + d[i] * e[i] (if a[i] < 0)
    do_second_op = (~a_nonneg) & mask
    b_update_vals = c_vals + d_vals * e_vals
    
    # Store updated a values
    tl.store(a_ptr + idx, new_a_vals, mask=mask)
    
    # Store updated b[i+1] values
    tl.store(b_ptr + b_next_idx, b_update_vals, mask=b_next_mask & do_second_op)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
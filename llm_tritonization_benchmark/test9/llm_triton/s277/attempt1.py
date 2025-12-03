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
    
    # Load b[i+1] for the computation
    b_next_mask = (idx + 1) < (n_elements + 1)
    b_next_vals = tl.load(b_ptr + idx + 1, mask=b_next_mask)
    
    # Compute conditions
    a_ge_zero = a_vals >= 0.0
    b_ge_zero = b_vals >= 0.0
    
    # Control flow logic:
    # if a[i] >= 0, skip everything (goto L20)
    # if b[i] >= 0, skip a[i] update but do b[i+1] update (goto L30)
    # otherwise, do a[i] update and b[i+1] update
    
    skip_all = a_ge_zero
    skip_a_update = b_ge_zero & (~skip_all)
    do_both = (~a_ge_zero) & (~b_ge_zero)
    do_b_only = skip_a_update | do_both
    
    # Update a[i] only when not skipping
    new_a_vals = tl.where(do_both, a_vals + c_vals * d_vals, a_vals)
    
    # Update b[i+1] when not skipping all
    new_b_next_vals = tl.where(do_b_only, c_vals + d_vals * e_vals, b_next_vals)
    
    # Store results
    tl.store(a_ptr + idx, new_a_vals, mask=mask)
    tl.store(b_ptr + idx + 1, new_b_next_vals, mask=b_next_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
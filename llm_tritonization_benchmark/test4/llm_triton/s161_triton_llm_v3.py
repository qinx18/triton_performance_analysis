import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Load c[i+1] for the second branch
        next_offsets = current_offsets + 1
        next_mask = mask & (next_offsets < n_elements + 1)
        c_next_vals = tl.load(c_ptr + next_offsets, mask=next_mask)
        
        # Condition: b[i] < 0
        condition = b_vals < 0.0
        
        # Branch 1: a[i] = c[i] + d[i] * e[i] (when b[i] >= 0)
        branch1_result = c_vals + d_vals * e_vals
        
        # Branch 2: c[i+1] = a[i] + d[i] * d[i] (when b[i] < 0)
        branch2_result = a_vals + d_vals * d_vals
        
        # Store a[i] for branch 1
        new_a_vals = tl.where(condition, a_vals, branch1_result)
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        
        # Store c[i+1] for branch 2
        new_c_next_vals = tl.where(condition, branch2_result, c_next_vals)
        tl.store(c_ptr + next_offsets, new_c_next_vals, mask=next_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    s161_kernel[(1,)](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
import triton
import triton.language as tl
import torch

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Load c[i+1] for the second branch
        c_next_mask = (current_offsets + 1) < n_elements
        c_next_vals = tl.load(c_ptr + current_offsets + 1, mask=c_next_mask)
        
        # Condition: b[i] < 0
        condition = b_vals < 0.0
        
        # Branch 1: a[i] = c[i] + d[i] * e[i]
        branch1_result = c_vals + d_vals * e_vals
        
        # Branch 2: c[i+1] = a[i] + d[i] * d[i]
        branch2_result = a_vals + d_vals * d_vals
        
        # Apply branch 1 (when b[i] >= 0)
        branch1_mask = mask & (~condition)
        tl.store(a_ptr + current_offsets, branch1_result, mask=branch1_mask)
        
        # Apply branch 2 (when b[i] < 0)
        branch2_mask = mask & condition & c_next_mask
        tl.store(c_ptr + current_offsets + 1, branch2_result, mask=branch2_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
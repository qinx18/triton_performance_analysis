import triton
import triton.language as tl
import torch

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Condition check: d[i] <= 0.0
        condition = d_vals <= 0.0
        
        # Compute both branches
        branch1 = b_vals * c_vals  # b[i] * c[i] when d[i] <= 0
        branch2 = b_vals * b_vals  # b[i] * b[i] when d[i] > 0
        
        # Select based on condition and update a
        result = tl.where(condition, branch1, branch2)
        new_a = a_vals + result
        
        # Store result
        tl.store(a_ptr + current_offsets, new_a, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    s443_kernel[(1,)](a, b, c, d, n_elements, BLOCK_SIZE)
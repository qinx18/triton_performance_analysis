import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Conditional computation: if (i+1 < mid)
        condition = (current_offsets + 1) < mid
        
        # Compute both cases
        case1 = b_vals * c_vals  # if condition is true
        case2 = b_vals * d_vals  # if condition is false
        
        # Select based on condition
        result = tl.where(condition, case1, case2)
        
        # Update a[i] += result
        new_a = a_vals + result
        
        # Store back to a
        tl.store(a_ptr + current_offsets, new_a, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s276_kernel[grid](
        a, b, c, d, n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE
    )
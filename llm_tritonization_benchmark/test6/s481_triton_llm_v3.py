import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load d values and check for negative values
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check if any d value is negative - if so, we need to handle exit condition
        has_negative = tl.sum((d_vals < 0.0).to(tl.int32), axis=0) > 0
        
        # For GPU implementation, we skip processing if negative found
        # (exit(0) equivalent - early termination)
        if has_negative:
            return
        
        # Load other arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] += b[i] * c[i]
        result = a_vals + b_vals * c_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
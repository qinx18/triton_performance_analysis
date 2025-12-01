import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load d values and check for negative values
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Check if any d value is negative
        has_negative = tl.any(d_vals < 0.0)
        if has_negative:
            # In GPU context, we can't exit the entire program
            # Instead, we'll skip computation for this block
            continue
        
        # Load other arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Compute a[i] += b[i] * c[i]
        result = a_vals + b_vals * c_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Check for negative values on CPU first to handle exit condition
    if torch.any(d < 0):
        return  # Equivalent to exit condition
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
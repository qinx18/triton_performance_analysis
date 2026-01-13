import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize s as a vector to maintain consistent types
    s = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process elements vectorized
        # Check condition and update s for any element where a > 0
        cond_mask = a_vals > 0.0
        d_squared = d_vals * d_vals
        s = tl.where(cond_mask, d_squared, s)
        
        # Compute results
        b_vals = s * c_vals + d_vals
        e_vals = (s + 1.0) * aa_vals
        
        # Store results
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        tl.store(e_ptr + current_offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s258_kernel[grid](
        a, aa, b, c, d, e, n_elements, BLOCK_SIZE
    )
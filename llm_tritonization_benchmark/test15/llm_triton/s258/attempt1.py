import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            offset = block_start + i
            
            # Load individual values for scalar operations
            a_val = tl.load(a_ptr + offset)
            d_val = tl.load(d_ptr + offset)
            c_val = tl.load(c_ptr + offset)
            aa_val = tl.load(aa_ptr + offset)
            
            # Update s if condition is met
            if a_val > 0.0:
                s = d_val * d_val
            
            # Compute and store results
            b_val = s * c_val + d_val
            e_val = (s + 1.0) * aa_val
            
            tl.store(b_ptr + offset, b_val)
            tl.store(e_ptr + offset, e_val)

def s258_triton(a, aa, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s258_kernel[grid](
        a, aa, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
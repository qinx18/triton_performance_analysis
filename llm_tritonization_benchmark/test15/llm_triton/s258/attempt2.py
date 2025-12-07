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
        
        # Load values for this block
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process elements in this block
        valid_count = tl.sum(mask.to(tl.int32))
        for i in range(BLOCK_SIZE):
            if i >= valid_count:
                continue
                
            # Check if a[i] > 0
            a_condition = a_vals[i] > 0.0
            if a_condition:
                s = d_vals[i] * d_vals[i]
            
            # Compute results
            b_val = s * c_vals[i] + d_vals[i]
            e_val = (s + 1.0) * aa_vals[i]
            
            # Store results
            offset = block_start + i
            if offset < n_elements:
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
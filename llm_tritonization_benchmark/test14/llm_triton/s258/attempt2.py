import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        current_offsets = i + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check condition and update s
        condition = a_vals > 0.0
        s_update = d_vals * d_vals
        
        # Update s for each valid element in the block
        for j in range(BLOCK_SIZE):
            element_idx = i + j
            if element_idx < n_elements:
                if tl.load(a_ptr + element_idx) > 0.0:
                    s = tl.load(d_ptr + element_idx) * tl.load(d_ptr + element_idx)
        
        # Compute b and e
        b_vals = s * c_vals + d_vals
        e_vals = (s + 1.0) * aa_vals
        
        # Store results
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        tl.store(e_ptr + current_offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use only the first row of aa (aa[0])
    aa_first_row = aa[0]
    
    # Launch kernel with single block to maintain sequential dependency
    grid = (1,)
    s258_kernel[grid](
        a, aa_first_row, b, c, d, e, n_elements, BLOCK_SIZE
    )
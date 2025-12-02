import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
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
            if block_start + i < n_elements:
                element_mask = tl.arange(0, BLOCK_SIZE) == i
                
                # Extract scalar values
                a_val = tl.sum(tl.where(element_mask, a_vals, 0.0))
                d_val = tl.sum(tl.where(element_mask, d_vals, 0.0))
                c_val = tl.sum(tl.where(element_mask, c_vals, 0.0))
                aa_val = tl.sum(tl.where(element_mask, aa_vals, 0.0))
                
                # Conditional update of s
                if a_val > 0.0:
                    s = d_val * d_val
                
                # Compute results
                b_result = s * c_val + d_val
                e_result = (s + 1.0) * aa_val
                
                # Store results
                idx = block_start + i
                tl.store(b_ptr + idx, b_result)
                tl.store(e_ptr + idx, e_result)

def s258_triton(a, aa, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (1,)
    
    s258_kernel[grid](
        a, aa, b, c, d, e, n_elements, BLOCK_SIZE
    )
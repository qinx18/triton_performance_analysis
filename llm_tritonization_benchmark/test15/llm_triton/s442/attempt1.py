import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        indx_vals = tl.load(indx_ptr + current_offsets, mask=mask)
        
        # Compute squares
        b_squared = b_vals * b_vals
        c_squared = c_vals * c_vals
        d_squared = d_vals * d_vals
        e_squared = e_vals * e_vals
        
        # Conditional computation based on indx values
        case1_mask = indx_vals == 1
        case2_mask = indx_vals == 2
        case3_mask = indx_vals == 3
        case4_mask = indx_vals == 4
        
        result = a_vals
        result = tl.where(case1_mask, result + b_squared, result)
        result = tl.where(case2_mask, result + c_squared, result)
        result = tl.where(case3_mask, result + d_squared, result)
        result = tl.where(case4_mask, result + e_squared, result)
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    s442_kernel[(1,)](
        a, b, c, d, e, indx, n_elements, BLOCK_SIZE
    )
    
    return a
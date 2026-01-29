import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load index values
        indx_vals = tl.load(indx_ptr + current_offsets, mask=mask, other=0)
        
        # Load array values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute based on switch/case logic
        case1_mask = (indx_vals == 1) & mask
        case2_mask = (indx_vals == 2) & mask
        case3_mask = (indx_vals == 3) & mask
        case4_mask = (indx_vals == 4) & mask
        
        # Apply computations based on cases
        result = a_vals
        result = tl.where(case1_mask, result + b_vals * b_vals, result)
        result = tl.where(case2_mask, result + c_vals * c_vals, result)
        result = tl.where(case3_mask, result + d_vals * d_vals, result)
        result = tl.where(case4_mask, result + e_vals * e_vals, result)
        
        # Store results
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    
    s442_kernel[(1,)](
        a, b, c, d, e, indx,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a
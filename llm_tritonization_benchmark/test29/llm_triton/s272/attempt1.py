import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Check condition
    condition = e_vals >= t
    
    # Only proceed if any element in the block satisfies the condition
    if tl.sum(condition.to(tl.int32)) > 0:
        a_vals = tl.load(a_ptr + idx, mask=mask)
        b_vals = tl.load(b_ptr + idx, mask=mask)
        c_vals = tl.load(c_ptr + idx, mask=mask)
        d_vals = tl.load(d_ptr + idx, mask=mask)
        
        # Apply conditional updates
        cd_product = c_vals * d_vals
        c_squared = c_vals * c_vals
        
        # Update only where condition is true
        a_new = tl.where(condition, a_vals + cd_product, a_vals)
        b_new = tl.where(condition, b_vals + c_squared, b_vals)
        
        # Store results
        tl.store(a_ptr + idx, a_new, mask=mask)
        tl.store(b_ptr + idx, b_new, mask=mask)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e,
        t,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
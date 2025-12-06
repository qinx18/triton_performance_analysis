import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_prev_vals = tl.load(c_ptr + current_offsets - 1, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute
        t1 = a_vals + b_vals
        new_a_vals = t1 + c_prev_vals
        t2 = c_vals * d_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        tl.store(c_ptr + current_offsets, t2, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    
    s261_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, c
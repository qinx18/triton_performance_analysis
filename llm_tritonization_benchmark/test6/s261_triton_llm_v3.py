import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially from i=1 to n_elements-1
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current elements
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load c[i-1] values
        c_prev_offsets = current_offsets - 1
        c_prev_mask = (current_offsets >= 1) & mask
        c_prev_vals = tl.load(c_ptr + c_prev_offsets, mask=c_prev_mask, other=0.0)
        
        # Compute: t = a[i] + b[i]; a[i] = t + c[i-1]
        t1 = a_vals + b_vals
        new_a_vals = t1 + c_prev_vals
        
        # Compute: t = c[i] * d[i]; c[i] = t
        t2 = c_vals * d_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        tl.store(c_ptr + current_offsets, t2, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    s261_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
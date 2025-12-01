import torch
import triton
import triton.language as tl

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Compute updates
        new_a = b_vals + c_vals
        new_b = c_vals * e_vals
        new_d = a_vals * e_vals
        
        # Store results
        tl.store(a_ptr + current_offsets + 1, new_a, mask=mask)
        tl.store(b_ptr + current_offsets, new_b, mask=mask)
        tl.store(d_ptr + current_offsets, new_d, mask=mask)

def s3251_triton(a, b, c, d, e):
    n = a.size(0) - 1
    BLOCK_SIZE = 256
    
    s3251_kernel[(1,)](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
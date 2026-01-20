import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values for current iteration
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # First statement: a[i] = b[i] + c[i] * d[i]
        a_vals_new = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals_new, mask=mask)
        
        # Second statement: b[i] = a[i] + d[i] * e[i]
        b_vals_new = a_vals_new + d_vals * e_vals
        tl.store(b_ptr + current_offsets, b_vals_new, mask=mask)
        
        # Third statement: a[i] = b[i] + a[i+1] * d[i]
        # Load a[i+1] from read-only copy
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n + 1)  # Allow reading a[i+1] where i+1 <= n
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        a_vals_final = b_vals_new + a_next_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals_final, mask=mask)

def s243_triton(a, b, c, d, e):
    n = a.shape[0] - 1  # Loop runs from 0 to LEN_1D-2 (i < LEN_1D-1)
    
    # Create read-only copy of array a to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
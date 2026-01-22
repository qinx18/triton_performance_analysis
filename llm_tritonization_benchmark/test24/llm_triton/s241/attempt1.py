import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # First computation: a[i] = b[i] * c[i] * d[i]
        a_new = b_vals * c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # For second computation, need a[i+1] values
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n + 1)  # Allow reading a[n] which exists in original array
        
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask, other=0.0)
        
        # Second computation: b[i] = a[i] * a[i+1] * d[i]
        b_new = a_new * a_next_vals * d_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s241_triton(a, b, c, d):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE
    )
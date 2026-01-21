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
        
        # Compute a[i] = b[i] * c[i] * d[i]
        a_vals = b_vals * c_vals * d_vals
        
        # Store a[i]
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Load a_copy[i+1] for second computation
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n + 1)
        a_copy_next = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        # Compute b[i] = a[i] * a[i+1] * d[i]
        b_new_vals = a_vals * a_copy_next * d_vals
        
        # Store b[i]
        tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy of 'a' to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE
    )
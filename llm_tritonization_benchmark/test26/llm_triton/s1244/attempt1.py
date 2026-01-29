import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
        a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
        
        # Store a[i]
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Load a[i+1] from copy (for d[i] = a[i] + a[i+1])
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n + 1)
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        # Compute d[i] = a[i] + a[i+1]
        d_vals = a_vals + a_next_vals
        
        # Store d[i]
        tl.store(d_ptr + current_offsets, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, a_copy, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
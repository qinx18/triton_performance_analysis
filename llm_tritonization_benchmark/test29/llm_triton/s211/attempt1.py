import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get thread block
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (i from 1 to n_elements-3 inclusive)
    mask = (offsets >= 1) & (offsets < n_elements - 2)
    
    # Load data for computing b[i]
    b_copy_vals = tl.load(b_copy_ptr + offsets + 1, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute b[i] = b_copy[i + 1] - e[i] * d[i]
    b_vals = b_copy_vals - e_vals * d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # Load data for computing a[i+1]
    c_vals_next = tl.load(c_ptr + offsets + 1, mask=mask)
    d_vals_next = tl.load(d_ptr + offsets + 1, mask=mask)
    
    # Compute a[i+1] = b[i] + c[i+1] * d[i+1]
    a_vals_next = b_vals + c_vals_next * d_vals_next
    tl.store(a_ptr + offsets + 1, a_vals_next, mask=mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy of b to avoid WAR dependency
    b_copy = b.clone()
    
    # Prologue: compute a[1]
    a[1] = b[0] + c[1] * d[1]
    
    # Main parallel loop range: i from 1 to n_elements-3 inclusive
    loop_size = n_elements - 2
    
    if loop_size > 1:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(loop_size, BLOCK_SIZE),)
        
        s211_kernel[grid](
            a, b, b_copy, c, d, e, 
            n_elements, 
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    # Epilogue: compute b[n_elements-2]
    if n_elements >= 3:
        b[n_elements-2] = b_copy[n_elements-1] - e[n_elements-2] * d[n_elements-2]
import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (N - 1)
        mask_plus_one = (current_offsets + 1) < N
        
        # Load arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        a_copy_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        a_copy_plus_one = tl.load(a_copy_ptr + current_offsets + 1, mask=mask_plus_one)
        
        # First computation: a[i] = b[i] * c[i] * d[i]
        a_vals = b_vals * c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Second computation: b[i] = a[i] * a[i+1] * d[i]
        # Use the newly computed a[i] and read a[i+1] from copy
        b_vals = a_vals * a_copy_plus_one * d_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s241_triton(a, b, c, d):
    N = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s241_kernel[grid](
        a, a_copy, b, c, d, N, BLOCK_SIZE=BLOCK_SIZE
    )
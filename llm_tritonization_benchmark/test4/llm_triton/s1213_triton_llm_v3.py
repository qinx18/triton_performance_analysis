import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n - 1
        
        # Load values
        b_prev = tl.load(b_ptr + current_offsets - 1, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        a_next = tl.load(a_ptr + current_offsets + 1, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = b[i-1] + c[i]
        a_new = b_prev + c_vals
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Compute b[i] = a[i+1] * d[i]
        b_new = a_next * d_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s1213_kernel[(1,)](a, b, c, d, n, BLOCK_SIZE)
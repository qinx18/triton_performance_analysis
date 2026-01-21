import triton
import triton.language as tl
import torch

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load d and e values
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Compute b[i] = d[i] * e[i]
        b_vals = d_vals * e_vals
        
        # Store b values
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # Load c and current a values for s152s operation
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Compute a[i] += b[i] * c[i] (s152s operation)
        a_vals = a_vals + b_vals * c_vals
        
        # Store updated a values
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s152_kernel[grid](a, b, c, d, e, N, BLOCK_SIZE=BLOCK_SIZE)
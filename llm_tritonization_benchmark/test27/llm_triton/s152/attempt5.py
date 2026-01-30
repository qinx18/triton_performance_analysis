import triton
import triton.language as tl
import torch

@triton.jit
def s152_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        idx = block_start + offsets
        mask = idx < N
        
        # Load d and e values
        d_vals = tl.load(d_ptr + idx, mask=mask)
        e_vals = tl.load(e_ptr + idx, mask=mask)
        
        # Compute b[i] = d[i] * e[i]
        b_vals = d_vals * e_vals
        
        # Store b values
        tl.store(b_ptr + idx, b_vals, mask=mask)
        
        # Load a and c values
        a_vals = tl.load(a_ptr + idx, mask=mask)
        c_vals = tl.load(c_ptr + idx, mask=mask)
        
        # Compute a[i] += b[i] * c[i]
        a_vals = a_vals + b_vals * c_vals
        
        # Store updated a values
        tl.store(a_ptr + idx, a_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    
    s152_kernel[(1,)](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
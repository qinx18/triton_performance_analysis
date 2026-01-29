import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < (n - 1)
    
    # Load arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_next_vals = tl.load(a_ptr + idx + 1, mask=mask)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new = c_vals + b_vals
    tl.store(b_ptr + idx, b_new, mask=mask)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i]
    a_next_new = b_new + a_next_vals * d_vals
    tl.store(a_ptr + idx + 1, a_next_new, mask=mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Copy arrays to preserve original values
    a_copy = a.clone()
    b_copy = b.clone()
    
    num_blocks = triton.cdiv(n - 1, BLOCK_SIZE)
    
    # Execute kernel for iterations 0 to n-2 (handling all dependencies sequentially)
    for i in range(0, n - 1):
        s244_kernel[(1,)](
            a_copy[i:], b_copy[i:], c[i:], d[i:], 
            1, BLOCK_SIZE=1
        )
    
    a.copy_(a_copy)
    b.copy_(b_copy)
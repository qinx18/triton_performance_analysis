import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n
    
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
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] (only for last iteration)
    last_mask = mask & (idx == n - 1)
    a_next_new = b_new + a_next_vals * d_vals
    tl.store(a_ptr + idx + 1, a_next_new, mask=last_mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Process iterations 0 to n-2
    loop_n = n - 1
    if loop_n > 0:
        num_blocks = triton.cdiv(loop_n, BLOCK_SIZE)
        
        s244_kernel[(num_blocks,)](
            a, b, c, d, loop_n, BLOCK_SIZE=BLOCK_SIZE
        )
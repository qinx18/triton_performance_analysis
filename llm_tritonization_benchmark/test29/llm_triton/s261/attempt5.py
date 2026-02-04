import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n
    
    # Load current values
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    
    # Load c[i-1] values
    c_prev_idx = idx - 1
    c_prev_mask = mask & (c_prev_idx >= 0)
    c_prev_vals = tl.load(c_ptr + c_prev_idx, mask=c_prev_mask, other=0.0)
    
    # Compute: t = a[i] + b[i]; a[i] = t + c[i-1]
    t1 = a_vals + b_vals
    new_a = t1 + c_prev_vals
    
    # Compute: t = c[i] * d[i]; c[i] = t
    new_c = c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(c_ptr + idx, new_c, mask=mask)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    
    if n <= 1:
        return a, c
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n - 1, BLOCK_SIZE)
    
    s261_kernel[(grid_size,)](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c
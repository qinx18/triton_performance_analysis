import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, b_copy_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < N - 1)
    
    # First statement: a[i] = b[i - 1] + c[i] * d[i]
    b_prev = tl.load(b_copy_ptr + offsets - 1, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    a_vals = b_prev + c_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Second statement: b[i] = b[i + 1] - e[i] * d[i]
    b_next = tl.load(b_copy_ptr + offsets + 1, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    b_vals = b_next - e_vals * d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, c, d, e, b_copy,
        N, BLOCK_SIZE
    )
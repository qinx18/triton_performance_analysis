import torch
import triton
import triton.language as tl

@triton.jit
def s491_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < N
    
    # Load indices and data
    indices = tl.load(ip_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Scatter store to a[ip[i]]
    tl.store(a_ptr + indices, result, mask=mask)

def s491_triton(a, b, c, d, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s491_kernel[grid](a, b, c, d, ip, N, BLOCK_SIZE=BLOCK_SIZE)
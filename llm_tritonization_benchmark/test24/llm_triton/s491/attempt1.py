import triton
import triton.language as tl
import torch

@triton.jit
def s491_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load indices for scatter operation
    ip_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Load b, c, d values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Scatter store to a[ip[i]]
    for i in range(BLOCK_SIZE):
        if block_start + i < n:
            ip_idx = tl.load(ip_ptr + block_start + i)
            val = tl.load(b_ptr + block_start + i) + tl.load(c_ptr + block_start + i) * tl.load(d_ptr + block_start + i)
            tl.store(a_ptr + ip_idx, val)

def s491_triton(a, b, c, d, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s491_kernel[grid](a, b, c, d, ip, n, BLOCK_SIZE)
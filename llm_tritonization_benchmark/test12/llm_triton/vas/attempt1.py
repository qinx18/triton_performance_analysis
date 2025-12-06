import torch
import triton
import triton.language as tl

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices and values
    ip_offsets = tl.load(ip_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    # Scatter: a[ip[i]] = b[i]
    for i in range(BLOCK_SIZE):
        if (block_start + i) < n_elements:
            idx = tl.load(ip_ptr + block_start + i)
            val = tl.load(b_ptr + block_start + i)
            tl.store(a_ptr + idx, val)

def vas_triton(a, b, ip):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](a, b, ip, n_elements, BLOCK_SIZE)
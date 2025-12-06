import torch
import triton
import triton.language as tl

@triton.jit
def s491_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create mask for valid elements
    mask = idx < n_elements
    
    # Load indices from ip array
    ip_vals = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Load values from b, c, d arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Compute b[i] + c[i] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Scatter to a[ip[i]] - handle each element individually
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            ip_val = tl.load(ip_ptr + block_start + i)
            val = tl.load(b_ptr + block_start + i) + tl.load(c_ptr + block_start + i) * tl.load(d_ptr + block_start + i)
            tl.store(a_ptr + ip_val, val)

def s491_triton(a, b, c, d, ip):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s491_kernel[grid](
        a, b, c, d, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
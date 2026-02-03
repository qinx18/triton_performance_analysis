import triton
import triton.language as tl
import torch

@triton.jit
def s491_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for bounds checking
    mask = indices < n_elements
    
    # Load b[i], c[i], d[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Load indirect indices ip[i]
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute b[i] + c[i] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Scatter store to a[ip[i]] only for valid indices
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            idx = tl.load(ip_ptr + block_start + i)
            val = tl.load(b_ptr + block_start + i) + tl.load(c_ptr + block_start + i) * tl.load(d_ptr + block_start + i)
            tl.store(a_ptr + idx, val)

def s491_triton(a, b, c, d, ip):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s491_kernel[(grid_size,)](
        a, b, c, d, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
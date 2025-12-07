import torch
import triton
import triton.language as tl

@triton.jit
def s491_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load b, c, d values
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Load indirect indices
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Store to indirect addresses (scatter operation)
    for i in range(BLOCK_SIZE):
        if indices[i] < n_elements:
            idx = tl.load(ip_ptr + indices[i])
            val = tl.load(b_ptr + indices[i]) + tl.load(c_ptr + indices[i]) * tl.load(d_ptr + indices[i])
            tl.store(a_ptr + idx, val)

def s491_triton(a, b, c, d, ip):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s491_kernel[grid](
        a, b, c, d, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
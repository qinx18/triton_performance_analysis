import torch
import triton
import triton.language as tl

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load values from b array
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Load indices from ip array
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # For each valid element, store b[i] at position a[ip[i]]
    for i in range(BLOCK_SIZE):
        if (block_start + i) < n_elements:
            actual_idx = block_start + i
            ip_idx = tl.load(ip_ptr + actual_idx)
            b_val = tl.load(b_ptr + actual_idx)
            tl.store(a_ptr + ip_idx, b_val)

def vas_triton(a, b, ip):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
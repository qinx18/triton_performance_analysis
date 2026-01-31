import triton
import triton.language as tl
import torch

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load b values and indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    ip_vals = tl.load(ip_ptr + indices, mask=mask)
    
    # Scatter: a[ip[i]] = b[i]
    # For each valid element, store b[i] at position ip[i]
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            idx_offset = i
            scatter_idx = tl.load(ip_ptr + block_start + idx_offset)
            b_val = tl.load(b_ptr + block_start + idx_offset)
            tl.store(a_ptr + scatter_idx, b_val)

def vas_triton(a, b, ip):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
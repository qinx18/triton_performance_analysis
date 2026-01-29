import triton
import triton.language as tl
import torch

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load indices and values
    indices = tl.load(ip_ptr + idx, mask=mask)
    values = tl.load(b_ptr + idx, mask=mask)
    
    # Scatter operation: a[ip[i]] = b[i]
    tl.store(a_ptr + indices, values, mask=mask)

def vas_triton(a, b, ip):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
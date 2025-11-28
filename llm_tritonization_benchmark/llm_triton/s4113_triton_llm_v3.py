import triton
import triton.language as tl
import torch

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices
    indices = tl.load(ip_ptr + offsets, mask=mask)
    
    # Load c[i] (direct access)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Load b[ip[i]] (gather operation)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute result
    result = b_vals + c_vals
    
    # Store a[ip[i]] = result (scatter operation)
    tl.store(a_ptr + indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4113_kernel[grid](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s4113_kernel(
    a_ptr, b_ptr, c_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + idx, mask=mask)
    
    # Load b[ip[i]] using gather
    b_vals = tl.load(b_ptr + ip_indices, mask=mask)
    
    # Load c[i] directly
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute b[ip[i]] + c[i]
    result = b_vals + c_vals
    
    # Store to a[ip[i]] using scatter
    tl.store(a_ptr + ip_indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4113_kernel[grid](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a
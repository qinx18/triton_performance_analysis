import triton
import triton.language as tl
import torch

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Load values using gathered indices
    b_vals = tl.load(b_ptr + ip_indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Compute result
    result = b_vals + c_vals
    
    # Store result using scattered indices
    tl.store(a_ptr + ip_indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4113_kernel[grid](
        a, b, c, ip,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load direct indices and indirect indices
    ip_vals = tl.load(ip_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Load b values using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=mask)
    
    # Compute result
    result = b_vals + c_vals
    
    # Store result using indirect addressing
    tl.store(a_ptr + ip_vals, result, mask=mask)

def s4113_triton(a, b, c, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4113_kernel[grid](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
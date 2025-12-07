import torch
import triton
import triton.language as tl

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load indices
    indices = tl.load(ip_ptr + idx, mask=mask)
    
    # Load values using indirect addressing
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute result
    result = b_vals + c_vals
    
    # Store result using indirect addressing
    tl.store(a_ptr + indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    n_elements = c.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4113_kernel[grid](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
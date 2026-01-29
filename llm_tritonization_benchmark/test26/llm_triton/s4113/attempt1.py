import triton
import triton.language as tl
import torch

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask)
    
    # Load c values (direct access)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    
    # Load b values (indirect access)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute result
    result = b_vals + c_vals
    
    # Store to a (indirect access)
    tl.store(a_ptr + indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4113_kernel[grid](
        a, b, c, ip,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
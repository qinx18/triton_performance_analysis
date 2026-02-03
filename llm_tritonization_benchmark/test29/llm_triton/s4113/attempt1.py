import triton
import triton.language as tl
import torch

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load indices from ip array
    indices = tl.load(ip_ptr + idx, mask=mask)
    
    # Gather from b array using indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load from c array (direct indexing)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute result
    result = b_vals + c_vals
    
    # Scatter to a array using indices
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
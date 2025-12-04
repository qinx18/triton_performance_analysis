import torch
import triton
import triton.language as tl

@triton.jit
def s491_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load b, c, d values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Load indices
    indices = tl.load(ip_ptr + idx, mask=mask)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Store to scattered locations
    # Use atomic store to handle potential race conditions
    tl.atomic_add(a_ptr + indices, result, mask=mask)

def s491_triton(a, b, c, d, ip):
    n_elements = b.shape[0]
    
    # Zero out a array first since we're using atomic_add
    a.zero_()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s491_kernel[grid](
        a, b, c, d, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s173_kernel(a_ptr, b_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load a[i] and b[i]
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+k]
    output_idx = idx + k
    tl.store(a_ptr + output_idx, result, mask=mask)

def s173_triton(a, b, k):
    n_elements = a.size(0) // 2
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s173_kernel[grid](
        a, b, k, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
import triton
import triton.language as tl
import torch

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Check condition b[i] > 0
    condition = b_vals > 0.0
    combined_mask = mask & condition
    
    # Load a and c values only where needed
    a_vals = tl.load(a_ptr + indices, mask=combined_mask)
    c_vals = tl.load(c_ptr + indices, mask=combined_mask)
    
    # Compute a[i] += b[i] * c[i] where condition is true
    result = a_vals + b_vals * c_vals
    
    # Store result back to a
    tl.store(a_ptr + indices, result, mask=combined_mask)

def s271_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s271_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
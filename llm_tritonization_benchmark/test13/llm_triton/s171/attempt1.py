import torch
import triton
import triton.language as tl

@triton.jit
def s171_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    inc,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load b[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Calculate a indices: i * inc
    a_idx = idx * inc
    
    # Load a[i * inc]
    a_vals = tl.load(a_ptr + a_idx, mask=mask)
    
    # Compute a[i * inc] += b[i]
    result = a_vals + b_vals
    
    # Store back to a[i * inc]
    tl.store(a_ptr + a_idx, result, mask=mask)

def s171_triton(a, b, inc):
    n_elements = b.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s171_kernel[grid](
        a,
        b,
        n_elements,
        inc,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a
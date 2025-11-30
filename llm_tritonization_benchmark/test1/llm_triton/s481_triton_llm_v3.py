import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load d values to check for negative values
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Check if any d[i] < 0 - if so, we need to handle the exit condition
    # In GPU context, we can't exit the entire program, so we'll skip computation
    # for threads where d[i] < 0
    compute_mask = mask & (d >= 0.0)
    
    # Load arrays with compute mask
    a = tl.load(a_ptr + offsets, mask=compute_mask)
    b = tl.load(b_ptr + offsets, mask=compute_mask)
    c = tl.load(c_ptr + offsets, mask=compute_mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a + b * c
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=compute_mask)

def s481_triton(a, b, c, d):
    n_elements = a.numel()
    
    # Check if any element in d is negative on CPU first
    if torch.any(d < 0):
        # In the original C code, this would exit the program
        # For GPU implementation, we'll raise an exception
        raise RuntimeError("Negative value found in array d - equivalent to exit(0) in original code")
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a
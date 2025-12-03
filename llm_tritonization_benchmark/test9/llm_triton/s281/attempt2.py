import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load original a values using reverse indexing
    reverse_idx = n_elements - 1 - idx
    a_reverse = tl.load(a_ptr + reverse_idx, mask=mask)
    
    # Load b and c values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
    x_vals = a_reverse + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + idx, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + idx, x_vals, mask=mask)
    tl.store(x_ptr + idx, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create a copy of array a to preserve original values for reverse access
    a_copy = a.clone()
    
    s281_kernel[grid](
        a, b, c, x, n_elements, BLOCK_SIZE
    )
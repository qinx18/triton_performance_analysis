import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_val = b + c * c + b * b + c
    
    # Store a[i]
    tl.store(a_ptr + offsets, a_val, mask=mask)
    
    # Load a[i+1] for d[i] computation
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)  # We need to read up to n_elements
    a_plus_1 = tl.load(a_ptr + offsets_plus_1, mask=mask_plus_1)
    
    # Compute d[i] = a[i] + a[i+1]
    d_val = a_val + a_plus_1
    
    # Store d[i]
    tl.store(d_ptr + offsets, d_val, mask=mask)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-2 (inclusive)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, b, c, d, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
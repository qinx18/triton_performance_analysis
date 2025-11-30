import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute first statement: a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_val = b + c * c + b * b + c
    
    # Store a[i]
    tl.store(a_ptr + offsets, a_val, mask=mask)

@triton.jit
def s1244_second_kernel(a_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[i] and a[i+1]
    a_i = tl.load(a_ptr + offsets, mask=mask)
    a_i_plus_1 = tl.load(a_ptr + offsets + 1, mask=mask)
    
    # Compute d[i] = a[i] + a[i+1]
    d_val = a_i + a_i_plus_1
    
    # Store d[i]
    tl.store(d_ptr + offsets, d_val, mask=mask)

def s1244_triton(a, b, c, d):
    n_elements = len(a) - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # First kernel: compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    s1244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: compute d[i] = a[i] + a[i+1]
    s1244_second_kernel[grid](
        a, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
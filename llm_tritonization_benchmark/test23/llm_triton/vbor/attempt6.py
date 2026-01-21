import triton
import triton.language as tl
import torch

@triton.jit
def vbor_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a1 = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b1 = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c1 = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d1 = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e1 = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    f1 = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    
    # Store original values
    a1_orig = a1
    b1_orig = b1
    c1_orig = c1
    d1_orig = d1
    e1_orig = e1
    f1_orig = f1
    
    # Compute a1
    a1 = (a1_orig * b1_orig * c1_orig + a1_orig * b1_orig * d1_orig + 
          a1_orig * b1_orig * e1_orig + a1_orig * b1_orig * f1_orig +
          a1_orig * c1_orig * d1_orig + a1_orig * c1_orig * e1_orig + 
          a1_orig * c1_orig * f1_orig + a1_orig * d1_orig * e1_orig +
          a1_orig * d1_orig * f1_orig + a1_orig * e1_orig * f1_orig)
    
    # Compute b1
    b1 = (b1_orig * c1_orig * d1_orig + b1_orig * c1_orig * e1_orig + 
          b1_orig * c1_orig * f1_orig + b1_orig * d1_orig * e1_orig +
          b1_orig * d1_orig * f1_orig + b1_orig * e1_orig * f1_orig)
    
    # Compute c1
    c1 = (c1_orig * d1_orig * e1_orig + c1_orig * d1_orig * f1_orig + 
          c1_orig * e1_orig * f1_orig)
    
    # Compute d1
    d1 = d1_orig * e1_orig * f1_orig
    
    # Final result
    result = a1 * b1 * c1 * d1
    
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, b, c, d, e, aa, x,
        n, BLOCK_SIZE=BLOCK_SIZE
    )
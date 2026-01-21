import triton
import triton.language as tl
import torch

@triton.jit
def vbor_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load original values
    a_orig = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_orig = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_orig = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_orig = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_orig = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    f_orig = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    
    # Copy to local variables for computation
    a1 = a_orig
    b1 = b_orig
    c1 = c_orig
    d1 = d_orig
    e1 = e_orig
    f1 = f_orig
    
    # Compute new a1 using original values
    a1 = (a_orig * b_orig * c_orig + a_orig * b_orig * d_orig + a_orig * b_orig * e_orig + a_orig * b_orig * f_orig +
          a_orig * c_orig * d_orig + a_orig * c_orig * e_orig + a_orig * c_orig * f_orig + a_orig * d_orig * e_orig +
          a_orig * d_orig * f_orig + a_orig * e_orig * f_orig)
    
    # Compute new b1 using original values  
    b1 = (b_orig * c_orig * d_orig + b_orig * c_orig * e_orig + b_orig * c_orig * f_orig + b_orig * d_orig * e_orig +
          b_orig * d_orig * f_orig + b_orig * e_orig * f_orig)
    
    # Compute new c1 using original values
    c1 = c_orig * d_orig * e_orig + c_orig * d_orig * f_orig + c_orig * e_orig * f_orig
    
    # Compute new d1 using original values
    d1 = d_orig * e_orig * f_orig
    
    # Final result using computed values
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n = aa.shape[1]  # LEN_2D
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, b, c, d, e, aa, x, n, BLOCK_SIZE=BLOCK_SIZE
    )
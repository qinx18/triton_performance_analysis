import triton
import triton.language as tl
import torch

@triton.jit
def vbor_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, 
                n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load original values
    a1 = tl.load(a_ptr + offsets, mask=mask)
    b1 = tl.load(b_ptr + offsets, mask=mask)
    c1 = tl.load(c_ptr + offsets, mask=mask)
    d1 = tl.load(d_ptr + offsets, mask=mask)
    e1 = tl.load(e_ptr + offsets, mask=mask)
    f1 = tl.load(aa_ptr + offsets, mask=mask)  # aa[0][i]
    
    # Store original values for reuse
    a1_orig = a1
    b1_orig = b1
    c1_orig = c1
    d1_orig = d1
    e1_orig = e1
    f1_orig = f1
    
    # Compute a1 - expanding the full expression
    a1 = a1_orig * b1_orig * c1_orig + a1_orig * b1_orig * d1_orig + a1_orig * b1_orig * e1_orig + a1_orig * b1_orig * f1_orig + a1_orig * c1_orig * d1_orig + a1_orig * c1_orig * e1_orig + a1_orig * c1_orig * f1_orig + a1_orig * d1_orig * e1_orig + a1_orig * d1_orig * f1_orig + a1_orig * e1_orig * f1_orig
    
    # Compute b1 - expanding the full expression  
    b1 = b1_orig * c1_orig * d1_orig + b1_orig * c1_orig * e1_orig + b1_orig * c1_orig * f1_orig + b1_orig * d1_orig * e1_orig + b1_orig * d1_orig * f1_orig + b1_orig * e1_orig * f1_orig
    
    # Compute c1 - expanding the full expression
    c1 = c1_orig * d1_orig * e1_orig + c1_orig * d1_orig * f1_orig + c1_orig * e1_orig * f1_orig
    
    # Compute d1
    d1 = d1_orig * e1_orig * f1_orig
    
    # Final result
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n = aa.shape[1]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    vbor_kernel[grid](a, aa, b, c, d, e, x, n, BLOCK_SIZE=BLOCK_SIZE)
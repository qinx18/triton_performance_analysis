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
    a1_orig = tl.load(a_ptr + offsets, mask=mask)
    b1_orig = tl.load(b_ptr + offsets, mask=mask)
    c1_orig = tl.load(c_ptr + offsets, mask=mask)
    d1_orig = tl.load(d_ptr + offsets, mask=mask)
    e1_orig = tl.load(e_ptr + offsets, mask=mask)
    f1_orig = tl.load(aa_ptr + offsets, mask=mask)  # aa[0][i]
    
    # Compute a1 = 10 terms - each term uses the updated a1 value
    a1 = a1_orig * b1_orig * c1_orig
    a1 = a1 + a1_orig * b1_orig * d1_orig
    a1 = a1 + a1_orig * b1_orig * e1_orig
    a1 = a1 + a1_orig * b1_orig * f1_orig
    a1 = a1 + a1_orig * c1_orig * d1_orig
    a1 = a1 + a1_orig * c1_orig * e1_orig
    a1 = a1 + a1_orig * c1_orig * f1_orig
    a1 = a1 + a1_orig * d1_orig * e1_orig
    a1 = a1 + a1_orig * d1_orig * f1_orig
    a1 = a1 + a1_orig * e1_orig * f1_orig
    
    # Compute b1 = 6 terms - each term uses the updated b1 value
    b1 = b1_orig * c1_orig * d1_orig
    b1 = b1 + b1_orig * c1_orig * e1_orig
    b1 = b1 + b1_orig * c1_orig * f1_orig
    b1 = b1 + b1_orig * d1_orig * e1_orig
    b1 = b1 + b1_orig * d1_orig * f1_orig
    b1 = b1 + b1_orig * e1_orig * f1_orig
    
    # Compute c1 = 3 terms - each term uses the updated c1 value
    c1 = c1_orig * d1_orig * e1_orig
    c1 = c1 + c1_orig * d1_orig * f1_orig
    c1 = c1 + c1_orig * e1_orig * f1_orig
    
    # Compute d1 = 1 term
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
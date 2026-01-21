import triton
import triton.language as tl
import torch

@triton.jit
def vbor_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load values
    a1 = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b1 = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c1 = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d1 = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e1 = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    f1 = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    
    # Store original values for computation
    a_orig = a1
    b_orig = b1
    c_orig = c1
    d_orig = d1
    e_orig = e1
    f_orig = f1
    
    # Compute a1 = a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1
    ab = a_orig * b_orig
    ac = a_orig * c_orig
    ad = a_orig * d_orig
    ae = a_orig * e_orig
    af = a_orig * f_orig
    
    a1 = ab * c_orig + ab * d_orig + ab * e_orig + ab * f_orig + ac * d_orig + ac * e_orig + ac * f_orig + ad * e_orig + ad * f_orig + ae * f_orig
    
    # Compute b1 = b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1
    bc = b_orig * c_orig
    bd = b_orig * d_orig
    be = b_orig * e_orig
    bf = b_orig * f_orig
    
    b1 = bc * d_orig + bc * e_orig + bc * f_orig + bd * e_orig + bd * f_orig + be * f_orig
    
    # Compute c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
    cd = c_orig * d_orig
    ce = c_orig * e_orig
    cf = c_orig * f_orig
    
    c1 = cd * e_orig + cd * f_orig + ce * f_orig
    
    # Compute d1 = d1 * e1 * f1
    d1 = d_orig * e_orig * f_orig
    
    # Final result
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n = aa.shape[1]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, b, c, d, e, aa, x, n, BLOCK_SIZE=BLOCK_SIZE
    )
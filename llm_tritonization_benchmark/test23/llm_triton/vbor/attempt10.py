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
    a1_orig = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b1_orig = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c1_orig = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d1_orig = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e1_orig = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    f1_orig = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    
    # Store originals in local variables that match C code exactly
    a1 = a1_orig
    b1 = b1_orig
    c1 = c1_orig
    d1 = d1_orig
    e1 = e1_orig
    f1 = f1_orig
    
    # Exactly as in C code - sequential updates using original values
    a1 = a1_orig * b1_orig * c1_orig + a1_orig * b1_orig * d1_orig + a1_orig * b1_orig * e1_orig + a1_orig * b1_orig * f1_orig + a1_orig * c1_orig * d1_orig + a1_orig * c1_orig * e1_orig + a1_orig * c1_orig * f1_orig + a1_orig * d1_orig * e1_orig + a1_orig * d1_orig * f1_orig + a1_orig * e1_orig * f1_orig
    
    b1 = b1_orig * c1_orig * d1_orig + b1_orig * c1_orig * e1_orig + b1_orig * c1_orig * f1_orig + b1_orig * d1_orig * e1_orig + b1_orig * d1_orig * f1_orig + b1_orig * e1_orig * f1_orig
    
    c1 = c1_orig * d1_orig * e1_orig + c1_orig * d1_orig * f1_orig + c1_orig * e1_orig * f1_orig
    
    d1 = d1_orig * e1_orig * f1_orig
    
    # Final result
    result = a1 * b1 * c1 * d1
    
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n = aa.shape[1]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, b, c, d, e, aa[0], x,
        n, BLOCK_SIZE=BLOCK_SIZE
    )
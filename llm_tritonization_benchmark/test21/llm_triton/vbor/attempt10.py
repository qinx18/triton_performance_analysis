import triton
import triton.language as tl
import torch

@triton.jit
def vbor_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a1 = tl.load(a_ptr + offsets, mask=mask)
    b1 = tl.load(b_ptr + offsets, mask=mask)
    c1 = tl.load(c_ptr + offsets, mask=mask)
    d1 = tl.load(d_ptr + offsets, mask=mask)
    e1 = tl.load(e_ptr + offsets, mask=mask)
    f1 = tl.load(aa_ptr + offsets, mask=mask)
    
    # Compute a1 (10 terms)
    a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
          a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 +
          a1 * d1 * f1 + a1 * e1 * f1)
    
    # Compute b1 (6 terms)
    b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 +
          b1 * d1 * f1 + b1 * e1 * f1)
    
    # Compute c1 (3 terms)
    c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
    
    # Compute d1 (1 term)
    d1 = d1 * e1 * f1
    
    # Final result
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n_elements = aa.shape[1]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, b, c, d, e, aa, x,
        n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
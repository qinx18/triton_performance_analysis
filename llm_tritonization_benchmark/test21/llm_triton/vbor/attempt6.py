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
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    f_vals = tl.load(aa_ptr + offsets, mask=mask)
    
    # Store original values for computation
    a1 = a_vals
    b1 = b_vals
    c1 = c_vals
    d1 = d_vals
    e1 = e_vals
    f1 = f_vals
    
    # Compute a1 with all combinations
    a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
          a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 +
          a1 * d1 * f1 + a1 * e1 * f1)
    
    # Compute b1 with remaining combinations
    b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 +
          b1 * d1 * f1 + b1 * e1 * f1)
    
    # Compute c1 with remaining combinations
    c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
    
    # Compute d1
    d1 = d1 * e1 * f1
    
    # Final result
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n_elements = aa.shape[1]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, b, c, d, e, aa, x,
        n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
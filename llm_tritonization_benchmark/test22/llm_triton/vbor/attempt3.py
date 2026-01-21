import triton
import triton.language as tl
import torch

@triton.jit
def vbor_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, x_ptr, 
                n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load input values
    a1 = tl.load(a_ptr + current_offsets, mask=mask)
    b1 = tl.load(b_ptr + current_offsets, mask=mask)
    c1 = tl.load(c_ptr + current_offsets, mask=mask)
    d1 = tl.load(d_ptr + current_offsets, mask=mask)
    e1 = tl.load(e_ptr + current_offsets, mask=mask)
    f1 = tl.load(aa_ptr + current_offsets, mask=mask)
    
    # Compute a1 using original loaded values
    a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
          a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 +
          a1 * d1 * f1 + a1 * e1 * f1)
    
    # Compute b1 using original loaded values
    b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 +
          b1 * d1 * f1 + b1 * e1 * f1)
    
    # Compute c1 using original loaded values
    c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
    
    # Compute d1 using original loaded values
    d1 = d1 * e1 * f1
    
    # Final computation
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + current_offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, b, c, d, e, aa, x,
        N, BLOCK_SIZE=BLOCK_SIZE
    )
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
    
    # Load input values and keep original values for later computations
    a1_orig = tl.load(a_ptr + current_offsets, mask=mask)
    b1_orig = tl.load(b_ptr + current_offsets, mask=mask)
    c1_orig = tl.load(c_ptr + current_offsets, mask=mask)
    d1_orig = tl.load(d_ptr + current_offsets, mask=mask)
    e1_orig = tl.load(e_ptr + current_offsets, mask=mask)
    f1_orig = tl.load(aa_ptr + current_offsets, mask=mask)
    
    # Compute a1 using original loaded values
    a1 = (a1_orig * b1_orig * c1_orig + a1_orig * b1_orig * d1_orig + a1_orig * b1_orig * e1_orig + a1_orig * b1_orig * f1_orig +
          a1_orig * c1_orig * d1_orig + a1_orig * c1_orig * e1_orig + a1_orig * c1_orig * f1_orig + a1_orig * d1_orig * e1_orig +
          a1_orig * d1_orig * f1_orig + a1_orig * e1_orig * f1_orig)
    
    # Compute b1 using original loaded values
    b1 = (b1_orig * c1_orig * d1_orig + b1_orig * c1_orig * e1_orig + b1_orig * c1_orig * f1_orig + b1_orig * d1_orig * e1_orig +
          b1_orig * d1_orig * f1_orig + b1_orig * e1_orig * f1_orig)
    
    # Compute c1 using original loaded values
    c1 = c1_orig * d1_orig * e1_orig + c1_orig * d1_orig * f1_orig + c1_orig * e1_orig * f1_orig
    
    # Compute d1 using original loaded values
    d1 = d1_orig * e1_orig * f1_orig
    
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
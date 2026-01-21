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
    a_val = tl.load(a_ptr + current_offsets, mask=mask)
    b_val = tl.load(b_ptr + current_offsets, mask=mask)
    c_val = tl.load(c_ptr + current_offsets, mask=mask)
    d_val = tl.load(d_ptr + current_offsets, mask=mask)
    e_val = tl.load(e_ptr + current_offsets, mask=mask)
    f_val = tl.load(aa_ptr + current_offsets, mask=mask)  # aa[0][i]
    
    # Store original values for computation
    a_orig = a_val
    b_orig = b_val
    c_orig = c_val
    d_orig = d_val
    e_orig = e_val
    f_orig = f_val
    
    # Compute a1 terms using original values
    a1 = (a_orig * b_orig * c_orig + a_orig * b_orig * d_orig + a_orig * b_orig * e_orig + a_orig * b_orig * f_orig +
          a_orig * c_orig * d_orig + a_orig * c_orig * e_orig + a_orig * c_orig * f_orig + a_orig * d_orig * e_orig +
          a_orig * d_orig * f_orig + a_orig * e_orig * f_orig)
    
    # Compute b1 terms using original values
    b1 = (b_orig * c_orig * d_orig + b_orig * c_orig * e_orig + b_orig * c_orig * f_orig + b_orig * d_orig * e_orig +
          b_orig * d_orig * f_orig + b_orig * e_orig * f_orig)
    
    # Compute c1 terms using original values
    c1 = c_orig * d_orig * e_orig + c_orig * d_orig * f_orig + c_orig * e_orig * f_orig
    
    # Compute d1 terms using original values
    d1 = d_orig * e_orig * f_orig
    
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
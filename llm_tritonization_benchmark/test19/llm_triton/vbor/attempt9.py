import triton
import triton.language as tl

@triton.jit
def vbor_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load original values
    a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_val = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_val = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_val = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    f_val = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    
    # Store original values for consistent calculations
    a_orig = a_val
    b_orig = b_val
    c_orig = c_val
    d_orig = d_val
    e_orig = e_val
    f_orig = f_val
    
    # Calculate a1 using original values
    a1 = (a_orig * b_orig * c_orig + a_orig * b_orig * d_orig + a_orig * b_orig * e_orig + a_orig * b_orig * f_orig +
          a_orig * c_orig * d_orig + a_orig * c_orig * e_orig + a_orig * c_orig * f_orig + a_orig * d_orig * e_orig +
          a_orig * d_orig * f_orig + a_orig * e_orig * f_orig)
    
    # Calculate b1 using original values
    b1 = (b_orig * c_orig * d_orig + b_orig * c_orig * e_orig + b_orig * c_orig * f_orig + b_orig * d_orig * e_orig +
          b_orig * d_orig * f_orig + b_orig * e_orig * f_orig)
    
    # Calculate c1 using original values
    c1 = (c_orig * d_orig * e_orig + c_orig * d_orig * f_orig + c_orig * e_orig * f_orig)
    
    # Calculate d1 using original values
    d1 = d_orig * e_orig * f_orig
    
    # Calculate final result
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, aa, b, c, d, e, x,
        N, BLOCK_SIZE
    )
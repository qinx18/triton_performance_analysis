import triton
import triton.language as tl

@triton.jit
def vbor_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load values
    a1 = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b1 = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c1 = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d1 = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e1 = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    f1 = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    
    # Save original values for WAR dependency handling
    a_orig = a1
    b_orig = b1
    c_orig = c1
    d_orig = d1
    e_orig = e1
    f_orig = f1
    
    # Compute new a1 using all original values
    a1 = (a_orig * b_orig * c_orig + a_orig * b_orig * d_orig + a_orig * b_orig * e_orig + a_orig * b_orig * f_orig +
          a_orig * c_orig * d_orig + a_orig * c_orig * e_orig + a_orig * c_orig * f_orig + a_orig * d_orig * e_orig +
          a_orig * d_orig * f_orig + a_orig * e_orig * f_orig)
    
    # Compute new b1 using all original values
    b1 = (b_orig * c_orig * d_orig + b_orig * c_orig * e_orig + b_orig * c_orig * f_orig + b_orig * d_orig * e_orig +
          b_orig * d_orig * f_orig + b_orig * e_orig * f_orig)
    
    # Compute new c1 using all original values
    c1 = (c_orig * d_orig * e_orig + c_orig * d_orig * f_orig + c_orig * e_orig * f_orig)
    
    # Compute new d1 using all original values
    d1 = d_orig * e_orig * f_orig
    
    # Final result using all new values
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
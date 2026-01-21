import triton
import triton.language as tl

@triton.jit
def vbor_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, 
                n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a1 = tl.load(a_ptr + offsets, mask=mask)
    b1 = tl.load(b_ptr + offsets, mask=mask)  
    c1 = tl.load(c_ptr + offsets, mask=mask)
    d1 = tl.load(d_ptr + offsets, mask=mask)
    e1 = tl.load(e_ptr + offsets, mask=mask)
    f1 = tl.load(aa_ptr + offsets, mask=mask)
    
    # Store original values for reuse
    a_orig = a1
    b_orig = b1
    c_orig = c1
    d_orig = d1
    e_orig = e1
    f_orig = f1
    
    # Compute a1 = a_orig * b_orig * c_orig + a_orig * b_orig * d_orig + ...
    a1 = (a_orig * b_orig * c_orig + a_orig * b_orig * d_orig + a_orig * b_orig * e_orig + a_orig * b_orig * f_orig +
          a_orig * c_orig * d_orig + a_orig * c_orig * e_orig + a_orig * c_orig * f_orig + a_orig * d_orig * e_orig +
          a_orig * d_orig * f_orig + a_orig * e_orig * f_orig)
    
    # Compute b1 = b_orig * c_orig * d_orig + b_orig * c_orig * e_orig + ...
    b1 = (b_orig * c_orig * d_orig + b_orig * c_orig * e_orig + b_orig * c_orig * f_orig + b_orig * d_orig * e_orig +
          b_orig * d_orig * f_orig + b_orig * e_orig * f_orig)
    
    # Compute c1 = c_orig * d_orig * e_orig + c_orig * d_orig * f_orig + c_orig * e_orig * f_orig
    c1 = c_orig * d_orig * e_orig + c_orig * d_orig * f_orig + c_orig * e_orig * f_orig
    
    # Compute d1 = d_orig * e_orig * f_orig
    d1 = d_orig * e_orig * f_orig
    
    # Final computation and store
    result = a1 * b1 * c1 * d1
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n_elements = aa.shape[1]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, aa, b, c, d, e, x,
        n_elements, BLOCK_SIZE
    )
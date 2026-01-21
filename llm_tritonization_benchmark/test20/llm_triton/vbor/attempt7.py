import triton
import triton.language as tl

@triton.jit
def vbor_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, 
                n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load original values
    a1_orig = tl.load(a_ptr + offsets, mask=mask)
    b1_orig = tl.load(b_ptr + offsets, mask=mask)  
    c1_orig = tl.load(c_ptr + offsets, mask=mask)
    d1_orig = tl.load(d_ptr + offsets, mask=mask)
    e1_orig = tl.load(e_ptr + offsets, mask=mask)
    f1_orig = tl.load(aa_ptr + offsets, mask=mask)
    
    # First computation: a1 using original values
    a1 = (a1_orig * b1_orig * c1_orig + a1_orig * b1_orig * d1_orig + a1_orig * b1_orig * e1_orig + a1_orig * b1_orig * f1_orig +
          a1_orig * c1_orig * d1_orig + a1_orig * c1_orig * e1_orig + a1_orig * c1_orig * f1_orig + a1_orig * d1_orig * e1_orig +
          a1_orig * d1_orig * f1_orig + a1_orig * e1_orig * f1_orig)
    
    # Second computation: b1 using original values
    b1 = (b1_orig * c1_orig * d1_orig + b1_orig * c1_orig * e1_orig + b1_orig * c1_orig * f1_orig + b1_orig * d1_orig * e1_orig +
          b1_orig * d1_orig * f1_orig + b1_orig * e1_orig * f1_orig)
    
    # Third computation: c1 using original values
    c1 = c1_orig * d1_orig * e1_orig + c1_orig * d1_orig * f1_orig + c1_orig * e1_orig * f1_orig
    
    # Fourth computation: d1 using original values
    d1 = d1_orig * e1_orig * f1_orig
    
    # Final computation and store
    result = a1 * b1 * c1 * d1
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n_elements = aa.shape[1]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, aa, b, c, d, e, x,
        n_elements, BLOCK_SIZE
    )
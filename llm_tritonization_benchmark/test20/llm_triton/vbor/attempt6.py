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
    
    # First computation: a1
    a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
          a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 +
          a1 * d1 * f1 + a1 * e1 * f1)
    
    # Second computation: b1  
    b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 +
          b1 * d1 * f1 + b1 * e1 * f1)
    
    # Third computation: c1
    c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
    
    # Fourth computation: d1
    d1 = d1 * e1 * f1
    
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
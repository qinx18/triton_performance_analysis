import triton
import triton.language as tl

@triton.jit
def vbor_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load all values
    a1_orig = tl.load(a_ptr + block_start + offsets, mask=mask)
    b1_orig = tl.load(b_ptr + block_start + offsets, mask=mask)
    c1_orig = tl.load(c_ptr + block_start + offsets, mask=mask)
    d1_orig = tl.load(d_ptr + block_start + offsets, mask=mask)
    e1_orig = tl.load(e_ptr + block_start + offsets, mask=mask)
    f1 = tl.load(aa_ptr + block_start + offsets, mask=mask)
    
    # Compute a1 (redefine a1 with the complex expression)
    a1 = (a1_orig * b1_orig * c1_orig + a1_orig * b1_orig * d1_orig + a1_orig * b1_orig * e1_orig + a1_orig * b1_orig * f1 +
          a1_orig * c1_orig * d1_orig + a1_orig * c1_orig * e1_orig + a1_orig * c1_orig * f1 + a1_orig * d1_orig * e1_orig +
          a1_orig * d1_orig * f1 + a1_orig * e1_orig * f1)
    
    # Compute b1 (redefine b1 with combinations)
    b1 = (b1_orig * c1_orig * d1_orig + b1_orig * c1_orig * e1_orig + b1_orig * c1_orig * f1 + b1_orig * d1_orig * e1_orig +
          b1_orig * d1_orig * f1 + b1_orig * e1_orig * f1)
    
    # Compute c1 (redefine c1 with combinations)
    c1 = c1_orig * d1_orig * e1_orig + c1_orig * d1_orig * f1 + c1_orig * e1_orig * f1
    
    # Compute d1 (redefine d1)
    d1 = d1_orig * e1_orig * f1
    
    # Final result
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + block_start + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    n_elements = aa.shape[1]  # LEN_2D
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, b, c, d, e, aa, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
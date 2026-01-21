import triton
import triton.language as tl

@triton.jit
def vbor_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load values
    a1_orig = tl.load(a_ptr + indices, mask=mask)
    b1_orig = tl.load(b_ptr + indices, mask=mask)
    c1_orig = tl.load(c_ptr + indices, mask=mask)
    d1_orig = tl.load(d_ptr + indices, mask=mask)
    e1_orig = tl.load(e_ptr + indices, mask=mask)
    f1 = tl.load(aa_ptr + indices, mask=mask)  # aa[0][i]
    
    # Update a1: 10 terms
    a1 = (a1_orig * b1_orig * c1_orig + a1_orig * b1_orig * d1_orig + a1_orig * b1_orig * e1_orig + a1_orig * b1_orig * f1 +
          a1_orig * c1_orig * d1_orig + a1_orig * c1_orig * e1_orig + a1_orig * c1_orig * f1 + a1_orig * d1_orig * e1_orig +
          a1_orig * d1_orig * f1 + a1_orig * e1_orig * f1)
    
    # Update b1: 6 terms
    b1 = (b1_orig * c1_orig * d1_orig + b1_orig * c1_orig * e1_orig + b1_orig * c1_orig * f1 + b1_orig * d1_orig * e1_orig +
          b1_orig * d1_orig * f1 + b1_orig * e1_orig * f1)
    
    # Update c1: 3 terms
    c1 = c1_orig * d1_orig * e1_orig + c1_orig * d1_orig * f1 + c1_orig * e1_orig * f1
    
    # Update d1: 1 term
    d1 = d1_orig * e1_orig * f1
    
    # Final result
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + indices, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    N = aa.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, aa, b, c, d, e, x,
        N, BLOCK_SIZE
    )
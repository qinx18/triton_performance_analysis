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
    a1 = tl.load(a_ptr + indices, mask=mask)
    b1 = tl.load(b_ptr + indices, mask=mask)
    c1 = tl.load(c_ptr + indices, mask=mask)
    d1 = tl.load(d_ptr + indices, mask=mask)
    e1 = tl.load(e_ptr + indices, mask=mask)
    f1 = tl.load(aa_ptr + indices, mask=mask)
    
    # Perform computations exactly as in C code
    a1 = a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1
    b1 = b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1
    c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
    d1 = d1 * e1 * f1
    result = a1 * b1 * c1 * d1
    
    # Store result
    tl.store(x_ptr + indices, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    N = aa.shape[1]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, aa, b, c, d, e, x,
        N, BLOCK_SIZE
    )
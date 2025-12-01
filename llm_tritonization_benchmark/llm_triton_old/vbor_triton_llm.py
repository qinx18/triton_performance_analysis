import torch
import triton
import triton.language as tl

@triton.jit
def vbor_kernel(
    a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load input values with masking
    a1 = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b1 = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c1 = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d1 = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e1 = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    f1 = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    
    # Compute intermediate products to reduce redundant calculations
    ab = a1 * b1
    ac = a1 * c1
    ad = a1 * d1
    ae = a1 * e1
    af = a1 * f1
    bc = b1 * c1
    bd = b1 * d1
    be = b1 * e1
    bf = b1 * f1
    cd = c1 * d1
    ce = c1 * e1
    cf = c1 * f1
    de = d1 * e1
    df = d1 * f1
    ef = e1 * f1
    
    # Compute a1 using precomputed products
    a1_new = (ab * c1 + ab * d1 + ab * e1 + ab * f1 +
              ac * d1 + ac * e1 + ac * f1 + ad * e1 +
              ad * f1 + ae * f1)
    
    # Compute b1 using precomputed products
    b1_new = (bc * d1 + bc * e1 + bc * f1 + bd * e1 +
              bd * f1 + be * f1)
    
    # Compute c1 using precomputed products
    c1_new = cd * e1 + cd * f1 + ce * f1
    
    # Compute d1 using precomputed products
    d1_new = de * f1
    
    # Final result
    result = a1_new * b1_new * c1_new * d1_new
    
    # Store result with masking
    tl.store(x_ptr + offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x):
    """
    Triton implementation of TSVC vbor function.
    Optimized with precomputed intermediate products and efficient memory access.
    """
    n_elements = a.numel()
    
    # Use power-of-2 block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with aa[0] as the f1 input
    vbor_kernel[grid](
        a, aa[0], b, c, d, e, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x
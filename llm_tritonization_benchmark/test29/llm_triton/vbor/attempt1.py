import triton
import triton.language as tl
import torch

@triton.jit
def vbor_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a1 = tl.load(a_ptr + current_offsets, mask=mask)
        b1 = tl.load(b_ptr + current_offsets, mask=mask)
        c1 = tl.load(c_ptr + current_offsets, mask=mask)
        d1 = tl.load(d_ptr + current_offsets, mask=mask)
        e1 = tl.load(e_ptr + current_offsets, mask=mask)
        f1 = tl.load(aa_ptr + current_offsets, mask=mask)
        
        # Compute complex expressions
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
              a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 +
              a1 * d1 * f1 + a1 * e1 * f1)
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 +
              b1 * d1 * f1 + b1 * e1 * f1)
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        d1 = d1 * e1 * f1
        
        result = a1 * b1 * c1 * d1
        
        tl.store(x_ptr + current_offsets, result, mask=mask)

def vbor_triton(a, aa, b, c, d, e, x, len_2d):
    n_elements = len_2d
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vbor_kernel[grid](
        a, aa, b, c, d, e, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return x
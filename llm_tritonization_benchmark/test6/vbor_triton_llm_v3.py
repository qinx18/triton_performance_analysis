import torch
import triton
import triton.language as tl

@triton.jit
def vbor_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a1 = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b1 = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c1 = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d1 = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e1 = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        f1 = tl.load(aa_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute new a1 - 59 flops total
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
              a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 +
              a1 * d1 * f1 + a1 * e1 * f1)
        
        # Compute new b1
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 +
              b1 * d1 * f1 + b1 * e1 * f1)
        
        # Compute new c1
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        
        # Compute new d1
        d1 = d1 * e1 * f1
        
        # Final computation and store
        result = a1 * b1 * c1 * d1
        tl.store(x_ptr + current_offsets, result, mask=mask)

def vbor_triton(a, b, c, d, e, aa, x):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    
    vbor_kernel[(1,)](
        a, b, c, d, e, aa, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x
import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        aa_vals = tl.load(aa_ptr + current_offsets, mask=mask)
        
        # Update s where a[i] > 0
        condition = a_vals > 0.0
        s = tl.where(condition, d_vals * d_vals, s)
        
        # b[i] = s * c[i] + d[i]
        b_vals = s * c_vals + d_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # e[i] = (s + 1.) * aa[0][i]
        e_vals = (s + 1.0) * aa_vals
        tl.store(e_ptr + current_offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (1,)
    
    s258_kernel[grid](
        a, aa, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
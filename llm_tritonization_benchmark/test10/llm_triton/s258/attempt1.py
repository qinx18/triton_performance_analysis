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
        
        # Load arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            offset_mask = i < BLOCK_SIZE
            if offset_mask:
                a_val = tl.load(a_ptr + block_start + i)
                d_val = tl.load(d_ptr + block_start + i)
                c_val = tl.load(c_ptr + block_start + i)
                aa_val = tl.load(aa_ptr + block_start + i)
                
                # if (a[i] > 0.) s = d[i] * d[i];
                if a_val > 0.0:
                    s = d_val * d_val
                
                # b[i] = s * c[i] + d[i];
                b_val = s * c_val + d_val
                tl.store(b_ptr + block_start + i, b_val)
                
                # e[i] = (s + 1.) * aa[0][i];
                e_val = (s + 1.0) * aa_val
                tl.store(e_ptr + block_start + i, e_val)

def s258_triton(a, aa, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s258_kernel[grid](
        a, aa, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
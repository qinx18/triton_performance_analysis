import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
        
        # Check condition for each element
        condition_mask = a_vals > 0.0
        
        # For elements where condition is true, update s with d[i] * d[i]
        # Since s is scalar, we need to handle this sequentially within each block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            if i < BLOCK_SIZE:
                elem_mask = (block_start + i) < n_elements
                if elem_mask:
                    a_val = tl.load(a_ptr + block_start + i)
                    d_val = tl.load(d_ptr + block_start + i)
                    c_val = tl.load(c_ptr + block_start + i)
                    aa_val = tl.load(aa_ptr + block_start + i)
                    
                    if a_val > 0.0:
                        s = d_val * d_val
                    
                    # Compute b[i] and e[i]
                    b_val = s * c_val + d_val
                    e_val = (s + 1.0) * aa_val
                    
                    tl.store(b_ptr + block_start + i, b_val)
                    tl.store(e_ptr + block_start + i, e_val)

def s258_triton(a, b, c, d, e, aa):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    s258_kernel[(1,)](
        a, b, c, d, e, aa,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return b, e
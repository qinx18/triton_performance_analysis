import torch
import triton
import triton.language as tl

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Compute: a[i+1] = b[i] + c[i]
        a_next_vals = b_vals + c_vals
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n_elements + 1)
        tl.store(a_ptr + next_offsets, a_next_vals, mask=next_mask)
        
        # Compute: b[i] = c[i] * e[i]
        b_new_vals = c_vals * e_vals
        tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)
        
        # Compute: d[i] = a[i] * e[i] (using original a[i])
        d_vals = a_vals * e_vals
        tl.store(d_ptr + current_offsets, d_vals, mask=mask)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
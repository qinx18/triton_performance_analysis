import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < n_elements) & (offsets > 0)
    
    if tl.sum(mask.to(tl.int32)) > 0:
        # Load current elements
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        d_vals = tl.load(d_ptr + offsets, mask=mask)
        e_vals = tl.load(e_ptr + offsets, mask=mask)
        
        # Load b[i-1] values
        b_prev_offsets = offsets - 1
        b_prev_mask = (b_prev_offsets >= 0) & (offsets < n_elements) & (offsets > 0)
        b_prev_vals = tl.load(b_ptr + b_prev_offsets, mask=b_prev_mask)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_vals = b_prev_vals + c_vals * d_vals
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_vals = a_vals + c_vals * e_vals
        
        # Store results
        tl.store(a_ptr + offsets, a_vals, mask=mask)
        tl.store(b_ptr + offsets, b_vals, mask=mask)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s323_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
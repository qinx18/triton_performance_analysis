import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load all values for this block
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    
    # Process sequentially within the block
    s = 0.0
    for i in tl.static_range(BLOCK_SIZE):
        idx_mask = (block_start + i) < n_elements
        
        # a[i] = s * e[i]
        a_val = s * e_vals[i]
        
        # s = b[i] + c[i] (only update if valid index)
        s = tl.where(idx_mask, b_vals[i] + c_vals[i], s)
        
        # b[i] = a[i] + d[i]
        b_val = a_val + d_vals[i]
        
        # Store results if valid
        if idx_mask:
            tl.store(a_ptr + block_start + i, a_val)
            tl.store(b_ptr + block_start + i, b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b
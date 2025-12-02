import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process one block at a time
    for i in range(0, n_elements, BLOCK_SIZE):
        current_offsets = i + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process elements sequentially due to dependency on s
        s = 0.0
        for j in range(BLOCK_SIZE):
            if i + j < n_elements:
                # a[i] = s * e[i]
                a_val = s * e_vals[j]
                tl.store(a_ptr + i + j, a_val)
                
                # s = b[i] + c[i]
                s = b_vals[j] + c_vals[j]
                
                # b[i] = a[i] + d[i]
                b_val = a_val + d_vals[j]
                tl.store(b_ptr + i + j, b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        current_offsets = i + offsets
        mask = current_offsets < n_elements
        
        # Load e[i]
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load b[i] and c[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load d[i]
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for j in range(BLOCK_SIZE):
            if i + j < n_elements:
                # a[i] = s * e[i]
                a_val = s * tl.load(e_ptr + i + j)
                tl.store(a_ptr + i + j, a_val)
                
                # s = b[i] + c[i]
                s = tl.load(b_ptr + i + j) + tl.load(c_ptr + i + j)
                
                # b[i] = a[i] + d[i]
                b_val = a_val + tl.load(d_ptr + i + j)
                tl.store(b_ptr + i + j, b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
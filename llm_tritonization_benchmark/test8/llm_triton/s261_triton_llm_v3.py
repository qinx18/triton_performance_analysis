import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially within each block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= 1 and idx < n_elements:
            # Load current elements
            a_val = tl.load(a_ptr + idx)
            b_val = tl.load(b_ptr + idx)
            c_val = tl.load(c_ptr + idx)
            c_prev_val = tl.load(c_ptr + idx - 1)
            d_val = tl.load(d_ptr + idx)
            
            # t = a[i] + b[i]
            t = a_val + b_val
            
            # a[i] = t + c[i-1]
            new_a = t + c_prev_val
            tl.store(a_ptr + idx, new_a)
            
            # t = c[i] * d[i]
            t = c_val * d_val
            
            # c[i] = t
            tl.store(c_ptr + idx, t)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
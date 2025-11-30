import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Process elements sequentially to handle dependencies
    for i in range(1, n_elements):
        if pid == 0:  # Only first block processes to maintain dependencies
            if i < n_elements:
                # t = a[i] + b[i]
                a_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + i)
                t = a_val + b_val
                
                # a[i] = t + c[i-1]
                c_prev = tl.load(c_ptr + (i-1))
                new_a = t + c_prev
                tl.store(a_ptr + i, new_a)
                
                # t = c[i] * d[i]
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                t = c_val * d_val
                
                # c[i] = t
                tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Use single block to maintain sequential dependencies
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 4
    
    mask = idx < n_elements
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_prev_vals = tl.load(b_ptr + idx - 4, mask=mask)
    
    result = b_prev_vals + a_vals
    
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    num_strips = triton.cdiv(n_elements - 4, STRIP_SIZE)
    
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        remaining = min(STRIP_SIZE, n_elements - 4 - strip_start)
        
        if remaining > 0:
            s1221_kernel[(1,)](
                a, b, strip_start, n_elements, 
                BLOCK_SIZE=min(BLOCK_SIZE, remaining)
            )
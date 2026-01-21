import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, strip_start, STRIP_SIZE: tl.constexpr):
    offsets = tl.arange(0, STRIP_SIZE)
    idx = strip_start + 4 + offsets
    
    mask = idx < strip_start + 4 + STRIP_SIZE
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_prev_vals = tl.load(b_ptr + idx - 4, mask=mask)
    
    result = b_prev_vals + a_vals
    
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    N = a.shape[0]
    STRIP_SIZE = 4
    n_elements = N - 4
    n_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    for strip_id in range(n_strips):
        strip_start = strip_id * STRIP_SIZE
        remaining = min(STRIP_SIZE, n_elements - strip_start)
        
        if remaining > 0:
            s1221_kernel[(1,)](
                a, b, strip_start, 
                STRIP_SIZE=remaining
            )
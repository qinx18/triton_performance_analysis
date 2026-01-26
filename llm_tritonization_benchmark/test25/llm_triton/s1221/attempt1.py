import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(b_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 4
    
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx - 4, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    result = b_vals + a_vals
    
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    N = a.shape[0]
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    n_elements = N
    
    for strip_start in range(0, n_elements - 4, STRIP_SIZE):
        remaining = min(STRIP_SIZE, n_elements - 4 - strip_start)
        if remaining <= 0:
            break
            
        s1221_kernel[(1,)](
            b, a, 
            strip_start, n_elements,
            BLOCK_SIZE=min(BLOCK_SIZE, remaining)
        )
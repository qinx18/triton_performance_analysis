import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(a_ptr, b_ptr, strip_start, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 4
    
    mask = idx < n
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx - 4, mask=mask)
    
    result = b_vals + a_vals
    
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    n = a.shape[0]
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    n_elements = n - 4
    
    for strip_id in range(triton.cdiv(n_elements, STRIP_SIZE)):
        strip_start = strip_id * STRIP_SIZE
        remaining = min(STRIP_SIZE, n_elements - strip_start)
        
        if remaining > 0:
            s1221_kernel[(1,)](a, b, strip_start, n, BLOCK_SIZE=min(BLOCK_SIZE, remaining))
import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, strip_start, strip_len, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + 1 + offsets
    
    mask = offsets < strip_len
    
    prev_vals = tl.load(a_ptr + idx - 1, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    result = prev_vals + s1 + s2 + b_vals + c_vals + d_vals
    
    tl.store(a_ptr + idx, result, mask=mask)

def s242_triton(a, b, c, d, s1, s2):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    STRIP_SIZE = 1
    BLOCK_SIZE = 1
    
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_len = min(STRIP_SIZE, n_elements - strip_start)
        
        s242_kernel[(1,)](
            a, b, c, d,
            s1, s2, strip_start, strip_len,
            BLOCK_SIZE=BLOCK_SIZE
        )
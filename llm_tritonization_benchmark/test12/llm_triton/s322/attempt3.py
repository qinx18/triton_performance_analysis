import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, strip_start, strip_size, LEN_1D, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = (offsets < strip_size) & (idx < LEN_1D)
    
    a_curr = tl.load(a_ptr + idx, mask=mask)
    a_prev1 = tl.load(a_ptr + idx - 1, mask=mask)
    a_prev2 = tl.load(a_ptr + idx - 2, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    result = a_curr + a_prev1 * b_vals + a_prev2 * c_vals
    
    tl.store(a_ptr + idx, result, mask=mask)

def s322_triton(a, b, c):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 2
    
    if n_elements <= 0:
        return
    
    STRIP_SIZE = 2
    BLOCK_SIZE = 2
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    for strip_id in range(num_strips):
        strip_start = 2 + strip_id * STRIP_SIZE
        remaining = min(STRIP_SIZE, n_elements - strip_id * STRIP_SIZE)
        
        if remaining <= 0:
            break
            
        s322_kernel[(1,)](
            a, b, c,
            strip_start, remaining, LEN_1D,
            BLOCK_SIZE=BLOCK_SIZE
        )
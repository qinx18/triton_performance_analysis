import torch
import triton
import triton.language as tl

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
    n_elements = a.shape[0]
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    for strip_start in range(0, n_elements - 4, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements - 4)
        actual_block_size = strip_end - strip_start
        
        if actual_block_size > 0:
            s1221_kernel[(1,)](
                b, a, strip_start, n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for strip_start in range(4, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        
        idx = strip_start + offsets
        mask = idx < strip_end
        
        read_idx = idx - 4
        a_vals = tl.load(a_ptr + idx, mask=mask)
        b_vals = tl.load(b_ptr + read_idx, mask=mask)
        
        result = b_vals + a_vals
        
        tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    s1221_kernel[(1,)](
        a, b,
        n_elements,
        STRIP_SIZE=STRIP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Load values
    a_vals = tl.load(a_ptr + idx)
    b_vals = tl.load(b_ptr + idx - 4)
    
    # Compute and store
    result = b_vals + a_vals
    tl.store(b_ptr + idx, result)

def s1221_triton(a, b):
    N = a.shape[0]
    STRIP_SIZE = 4
    n_elements = N - 4
    
    # Process in sequential strips of size 4
    for strip_start in range(4, N, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, N)
        current_strip_size = strip_end - strip_start
        
        if current_strip_size > 0:
            grid = (1,)
            s1221_kernel[grid](a, b, strip_start, BLOCK_SIZE=current_strip_size)
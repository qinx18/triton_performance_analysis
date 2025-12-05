import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Read from b[idx - 4] and a[idx]
    b_vals = tl.load(b_ptr + idx - 4)
    a_vals = tl.load(a_ptr + idx)
    
    # Compute b[idx] = b[idx - 4] + a[idx]
    result = b_vals + a_vals
    
    # Store result
    tl.store(b_ptr + idx, result)

def s1221_triton(a, b):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 4
    STRIP_SIZE = 4
    
    # Process in strips sequentially
    for strip_start in range(4, LEN_1D, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, LEN_1D)
        current_strip_size = strip_end - strip_start
        
        if current_strip_size > 0:
            s1221_kernel[(1,)](
                a, b, strip_start,
                BLOCK_SIZE=current_strip_size
            )
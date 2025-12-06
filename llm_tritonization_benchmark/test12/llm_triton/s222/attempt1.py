import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes one strip (strip size = 1 due to RAW dependency)
    strip_id = tl.program_id(0)
    strip_start = strip_id * 1
    
    if strip_start >= n_elements:
        return
    
    # Load values for this strip
    idx = strip_start + 1  # +1 because loop starts from i=1
    
    # First operation: a[i] += b[i] * c[i]
    a_val = tl.load(a_ptr + idx)
    b_val = tl.load(b_ptr + idx)
    c_val = tl.load(c_ptr + idx)
    a_val += b_val * c_val
    tl.store(a_ptr + idx, a_val)
    
    # Second operation: e[i] = e[i-1] * e[i-1]
    e_prev = tl.load(e_ptr + idx - 1)
    e_result = e_prev * e_prev
    tl.store(e_ptr + idx, e_result)
    
    # Third operation: a[i] -= b[i] * c[i]
    a_val = tl.load(a_ptr + idx)
    a_val -= b_val * c_val
    tl.store(a_ptr + idx, a_val)

def s222_triton(a, b, c, e):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1  # Loop runs from i=1 to LEN_1D-1
    
    if n_elements <= 0:
        return
    
    # Due to RAW dependency, must process sequentially strip by strip
    num_strips = n_elements
    
    # Launch strips sequentially to maintain dependency
    for strip_start in range(num_strips):
        s222_kernel[(1,)](a, b, c, e, n_elements, BLOCK_SIZE=1)
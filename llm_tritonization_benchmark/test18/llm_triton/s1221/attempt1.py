import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 4
    
    # Check bounds
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_prev_vals = tl.load(b_ptr + idx - 4, mask=mask)
    
    # Compute and store
    result = b_prev_vals + a_vals
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    # Process elements from index 4 to n_elements-1 in strips
    for strip_start in range(0, n_elements - 4, STRIP_SIZE):
        current_strip_size = min(STRIP_SIZE, n_elements - 4 - strip_start)
        
        s1221_kernel[(1,)](
            a, b,
            strip_start,
            n_elements,
            BLOCK_SIZE=current_strip_size
        )
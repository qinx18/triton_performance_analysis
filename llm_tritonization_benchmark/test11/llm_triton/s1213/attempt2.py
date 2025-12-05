import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, strip_start, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 1
    
    mask = idx < n_elements
    
    # First statement: a[i] = b[i-1] + c[i]
    b_vals = tl.load(b_ptr + idx - 1, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    a_result = b_vals + c_vals
    tl.store(a_ptr + idx, a_result, mask=mask)
    
    # Second statement: b[i] = a[i+1] * d[i] (use read-only copy)
    a_vals = tl.load(a_copy_ptr + idx + 1, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    b_result = a_vals * d_vals
    tl.store(b_ptr + idx, b_result, mask=mask)

def s1213_triton(a, b, c, d):
    # Create read-only copy of array a
    a_copy = a.clone()
    
    # Loop range: i from 1 to LEN_1D-2 (inclusive)
    n_elements = a.shape[0] - 1
    STRIP_SIZE = 1
    BLOCK_SIZE = 1
    
    num_strips = triton.cdiv(n_elements - 1, STRIP_SIZE)
    
    # Process strips sequentially due to loop-carried dependency
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        s1213_kernel[(1,)](
            a, a_copy, b, c, d, 
            strip_start,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
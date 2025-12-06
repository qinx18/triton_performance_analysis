import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes one strip sequentially
    strip_id = tl.program_id(0)
    strip_start = strip_id * 1
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid indices (1 <= idx < n_elements-1)
    mask = (idx >= 1) & (idx < n_elements - 1)
    
    # First statement: a[i] = b[i-1] + c[i]
    b_vals = tl.load(b_ptr + idx - 1, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    a_result = b_vals + c_vals
    tl.store(a_ptr + idx, a_result, mask=mask)
    
    # Second statement: b[i] = a[i+1] * d[i] (read from copy)
    a_vals = tl.load(a_copy_ptr + idx + 1, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    b_result = a_vals * d_vals
    tl.store(b_ptr + idx, b_result, mask=mask)

def s1213_triton(a, b, c, d):
    # Create read-only copy of a
    a_copy = a.clone()
    
    n_elements = a.size(0)
    STRIP_SIZE = 1
    BLOCK_SIZE = 1
    
    # Calculate number of strips
    num_strips = triton.cdiv(n_elements - 2, STRIP_SIZE)  # -2 because we process from 1 to n-2
    
    # Process strips sequentially due to RAW dependency
    for strip_id in range(num_strips):
        s1213_kernel[(1,)](
            a, a_copy, b, c, d,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
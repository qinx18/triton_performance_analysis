import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, strip_start, strip_size, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = (offsets < strip_size) & (idx < n_elements)
    
    # Load values
    a_curr = tl.load(a_ptr + idx, mask=mask)
    a_prev1 = tl.load(a_ptr + idx - 1, mask=mask)
    a_prev2 = tl.load(a_ptr + idx - 2, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
    result = a_curr + a_prev1 * b_vals + a_prev2 * c_vals
    
    tl.store(a_ptr + idx, result, mask=mask)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    STRIP_SIZE = 2
    BLOCK_SIZE = 2
    
    # Process strips sequentially from index 2 to n_elements
    strip_start = 2
    while strip_start < n_elements:
        current_strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        s322_kernel[(1,)](
            a, b, c,
            strip_start, current_strip_size, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        strip_start += STRIP_SIZE
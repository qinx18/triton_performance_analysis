import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, strip_start, n_elements, STRIP_SIZE: tl.constexpr):
    offsets = tl.arange(0, STRIP_SIZE)
    idx = strip_start + offsets + 4  # +4 because loop starts from i=4
    
    mask = idx < n_elements
    
    # Read from b[i-4] and a[i]
    b_vals = tl.load(b_ptr + idx - 4, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute b[i] = b[i-4] + a[i]
    result = b_vals + a_vals
    
    # Store result
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    N = a.shape[0]
    STRIP_SIZE = 4
    n_elements = N - 4  # Elements to process (from i=4 to N-1)
    
    # Process strips sequentially to handle loop-carried dependency
    for strip_start in range(0, n_elements, STRIP_SIZE):
        remaining = min(STRIP_SIZE, n_elements - strip_start)
        if remaining <= 0:
            break
            
        # Launch kernel for this strip
        grid = (1,)
        s1221_kernel[grid](a, b, strip_start, N, STRIP_SIZE=remaining)
import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(b_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + 4 + offsets
    
    mask = idx < n_elements
    
    # Read from b[i-4] and a[i]
    b_vals = tl.load(b_ptr + idx - 4, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute b[i] = b[i-4] + a[i]
    result = b_vals + a_vals
    
    # Store result
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    start_idx = 4
    total_elements = n_elements - start_idx
    
    if total_elements <= 0:
        return
    
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    # Process in strips of 4 elements sequentially
    for strip_start in range(0, total_elements, STRIP_SIZE):
        remaining = min(STRIP_SIZE, total_elements - strip_start)
        if remaining <= 0:
            break
            
        # Launch kernel for this strip
        s1221_kernel[(1,)](
            b, a,
            strip_start,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
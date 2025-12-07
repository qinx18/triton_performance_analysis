import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, strip_start, LEN_1D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 2
    
    mask = idx < LEN_1D
    
    # Load values - ensure we load from correct positions
    a_current = tl.load(a_ptr + idx, mask=mask, other=0.0)
    a_minus_1 = tl.load(a_ptr + (idx - 1), mask=(idx - 1) >= 0, other=0.0)
    a_minus_2 = tl.load(a_ptr + (idx - 2), mask=(idx - 2) >= 0, other=0.0)
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    
    # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
    result = a_current + a_minus_1 * b_vals + a_minus_2 * c_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s322_triton(a, b, c):
    LEN_1D = a.shape[0]
    STRIP_SIZE = 2
    BLOCK_SIZE = 2
    
    # Calculate number of elements to process (starting from index 2)
    n_elements = LEN_1D - 2
    
    # Process in strips sequentially - each strip has up to STRIP_SIZE elements
    for strip_start in range(0, n_elements, STRIP_SIZE):
        remaining = min(STRIP_SIZE, n_elements - strip_start)
        grid = (1,)
        
        # Launch kernel for this strip
        s322_kernel[grid](
            a, b, c,
            strip_start,
            LEN_1D,
            BLOCK_SIZE
        )
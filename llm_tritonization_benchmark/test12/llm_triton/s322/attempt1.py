import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes one strip of size 2
    strip_id = tl.program_id(0)
    strip_start = strip_id * 2 + 2  # Start from index 2
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid elements
    mask = idx < n_elements
    
    # Load a[i], a[i-1], a[i-2], b[i], c[i]
    a_curr = tl.load(a_ptr + idx, mask=mask)
    a_prev1 = tl.load(a_ptr + idx - 1, mask=mask)
    a_prev2 = tl.load(a_ptr + idx - 2, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
    result = a_curr + a_prev1 * b_vals + a_prev2 * c_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s322_triton(a, b, c):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 2  # Elements to process (from index 2 to LEN_1D-1)
    
    if n_elements <= 0:
        return
    
    STRIP_SIZE = 2
    BLOCK_SIZE = 2
    
    # Calculate number of strips needed
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially (each strip must complete before next starts)
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        remaining = min(STRIP_SIZE, n_elements - strip_start)
        
        if remaining <= 0:
            break
            
        # Launch kernel for this strip
        s322_kernel[(1,)](
            a, b, c,
            strip_start + remaining + 2,  # n_elements for this strip
            BLOCK_SIZE=min(BLOCK_SIZE, remaining)
        )
import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    strip_id = tl.program_id(0)
    strip_start = strip_id * 2 + 2  # Start from index 2
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < n_elements
    
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
    LEN_1D = a.shape[0]
    n_elements = LEN_1D
    
    STRIP_SIZE = 2
    BLOCK_SIZE = min(STRIP_SIZE, 2)  # Max block size is 2
    
    num_strips = triton.cdiv(n_elements - 2, STRIP_SIZE)
    
    # Process strips sequentially to maintain dependencies
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE + 2
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        
        if strip_start >= n_elements:
            break
            
        current_block_size = min(BLOCK_SIZE, strip_end - strip_start)
        
        s322_kernel[(1,)](
            a, b, c,
            n_elements,
            BLOCK_SIZE=current_block_size
        )
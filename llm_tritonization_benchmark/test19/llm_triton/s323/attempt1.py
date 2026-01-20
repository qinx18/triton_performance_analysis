import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < n_elements) & (offsets >= 1)
    
    # This is a strictly sequential pattern due to loop-carried dependency
    # Each iteration needs b[i-1] from the previous iteration
    # Process sequentially within each block
    for idx in range(BLOCK_SIZE):
        i = block_start + idx
        if i >= 1 and i < n_elements:
            # Load b[i-1], c[i], d[i], e[i]
            b_prev = tl.load(b_ptr + (i - 1))
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            
            # Compute a[i] = b[i-1] + c[i] * d[i]
            a_val = b_prev + c_val * d_val
            
            # Store a[i]
            tl.store(a_ptr + i, a_val)
            
            # Compute b[i] = a[i] + c[i] * e[i]
            b_val = a_val + c_val * e_val
            
            # Store b[i]
            tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s323_kernel[grid](
        a, b, c, d, e, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
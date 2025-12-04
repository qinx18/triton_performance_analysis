import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the sequential dependency in b[i] = b[i-1] + a[i] + d[i]
    # We process one element at a time to maintain the dependency
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially from index 1 to n_elements-1
    for i in range(1, n_elements):
        if i < n_elements:
            # Load c[i] and d[i]
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # Load a[i]
            a_val = tl.load(a_ptr + i)
            
            # Update a[i] += c[i] * d[i]
            a_new = a_val + c_val * d_val
            tl.store(a_ptr + i, a_new)
            
            # Load b[i-1] and b[i] for the update
            b_prev = tl.load(b_ptr + i - 1)
            
            # Update b[i] = b[i-1] + a[i] + d[i]
            b_new = b_prev + a_new + d_val
            tl.store(b_ptr + i, b_new)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Use BLOCK_SIZE of 1 since we need sequential processing
    BLOCK_SIZE = 1
    
    # Launch kernel with single block since we process sequentially
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
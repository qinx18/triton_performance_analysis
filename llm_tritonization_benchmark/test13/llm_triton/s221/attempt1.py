import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially due to loop-carried dependency
    for strip_start in range(0, n_elements, 1):
        strip_end = min(strip_start + 1, n_elements)
        strip_len = strip_end - strip_start
        
        if strip_len > 0:
            idx = strip_start + offsets[:strip_len]
            mask = offsets[:strip_len] < strip_len
            
            # Load values for a[i] += c[i] * d[i]
            a_vals = tl.load(a_ptr + idx + 1, mask=mask)
            c_vals = tl.load(c_ptr + idx + 1, mask=mask)
            d_vals = tl.load(d_ptr + idx + 1, mask=mask)
            
            # Update a[i]
            a_new = a_vals + c_vals * d_vals
            tl.store(a_ptr + idx + 1, a_new, mask=mask)
            
            # Load values for b[i] = b[i-1] + a[i] + d[i]
            b_prev = tl.load(b_ptr + idx, mask=mask)  # b[i-1]
            
            # Compute b[i]
            b_new = b_prev + a_new + d_vals
            tl.store(b_ptr + idx + 1, b_new, mask=mask)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop from 1 to LEN_1D-1
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 1  # Must be 1 due to loop-carried dependency
    
    # Launch single thread group since we need sequential processing
    s221_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be executed with a single thread block due to WAR dependency
    # b[i] = b[i-1] + a[i] + d[i] creates a sequential dependency
    
    # Only thread 0 performs the computation
    if tl.program_id(0) == 0:
        for i in range(1, n_elements):
            # Load values
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_val = tl.load(a_ptr + i)
            b_prev = tl.load(b_ptr + i - 1)
            
            # Compute a[i] += c[i] * d[i]
            new_a = a_val + c_val * d_val
            tl.store(a_ptr + i, new_a)
            
            # Compute b[i] = b[i-1] + a[i] + d[i]
            new_b = b_prev + new_a + d_val
            tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with single program to handle sequential dependency
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
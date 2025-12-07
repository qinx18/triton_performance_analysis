import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements):
    # This is a strictly sequential kernel - only one thread processes all elements
    pid = tl.program_id(0)
    
    # Only the first thread does the work
    if pid != 0:
        return
    
    # Process sequentially from i=2 to n_elements-1
    for i in range(2, n_elements):
        # Load values one by one
        a_i = tl.load(a_ptr + i)
        a_i_minus_1 = tl.load(a_ptr + (i - 1))
        a_i_minus_2 = tl.load(a_ptr + (i - 2))
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result back to a[i]
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Launch with only 1 block since this must be sequential
    grid = (1,)
    
    s322_kernel[grid](
        a, b, c,
        n_elements
    )
    
    return a
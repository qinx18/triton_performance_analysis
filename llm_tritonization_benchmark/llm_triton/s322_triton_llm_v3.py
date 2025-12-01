import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the recurrence sequentially since each element depends on previous ones
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    # Sequential processing from index 2 to n_elements-1
    for i in range(2, n_elements):
        # Load current values
        a_curr = tl.load(a_ptr + i)
        a_prev1 = tl.load(a_ptr + i - 1)
        a_prev2 = tl.load(a_ptr + i - 2)
        b_curr = tl.load(b_ptr + i)
        c_curr = tl.load(c_ptr + i)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
        
        # Store result
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    # Launch single thread since this is a sequential recurrence
    grid = (1,)
    
    s322_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
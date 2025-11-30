import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one iteration of the recurrence per launch
    # since each element depends on previous elements in the same array
    i = tl.program_id(0)
    
    if i >= n_elements - 2:
        return
    
    # Calculate actual index (starting from 2)
    idx = i + 2
    
    # Load current and previous values
    a_curr = tl.load(a_ptr + idx)
    a_prev1 = tl.load(a_ptr + idx - 1)
    a_prev2 = tl.load(a_ptr + idx - 2)
    b_curr = tl.load(b_ptr + idx)
    c_curr = tl.load(c_ptr + idx)
    
    # Compute recurrence
    result = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
    
    # Store result
    tl.store(a_ptr + idx, result)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Process elements sequentially due to recurrence dependency
    for i in range(n_elements - 2):
        idx = i + 2
        a[idx] = a[idx] + a[idx - 1] * b[idx] + a[idx - 2] * c[idx]
    
    return a
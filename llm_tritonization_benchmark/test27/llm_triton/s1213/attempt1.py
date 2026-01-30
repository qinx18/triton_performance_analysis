import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Use single thread to process sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n_elements-2
    for i in range(1, n_elements - 1):
        # Load b[i-1], c[i], a[i+1], d[i]
        b_prev = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        a_next = tl.load(a_ptr + i + 1)
        d_val = tl.load(d_ptr + i)
        
        # a[i] = b[i-1] + c[i]
        a_val = b_prev + c_val
        tl.store(a_ptr + i, a_val)
        
        # b[i] = a[i+1] * d[i]
        b_val = a_next * d_val
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy of array 'a' to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block for sequential execution
    
    s1213_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )
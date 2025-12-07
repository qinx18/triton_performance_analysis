import triton
import triton.language as tl
import torch

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be executed by a single thread due to strict sequential dependency
    pid = tl.program_id(0)
    
    # Only the first thread processes the entire array sequentially
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n_elements-1
    for i in range(1, n_elements):
        # Load values for a[i] operations
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # a[i] += b[i] * c[i]
        a_val = a_val + b_val * c_val
        tl.store(a_ptr + i, a_val)
        
        # e[i] = e[i - 1] * e[i - 1] (sequential dependency)
        e_prev = tl.load(e_ptr + i - 1)
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # a[i] -= b[i] * c[i]
        a_val = a_val - b_val * c_val
        tl.store(a_ptr + i, a_val)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Use only 1 thread due to sequential dependency
    grid = (1,)
    
    s222_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, e
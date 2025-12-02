import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles WAR dependency by processing each element sequentially
    # Cannot parallelize due to a[i+1] and a[i] dependencies
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    # Process elements sequentially to handle dependencies
    for i in range(n_elements - 1):
        # Load values
        b_val = tl.load(b_ptr + i)
        e_val = tl.load(e_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # First assignment: a[i+1] = b[i] + e[i]
        tl.store(a_ptr + i + 1, b_val + e_val)
        
        # Second assignment: a[i] = b[i] + c[i]
        tl.store(a_ptr + i, b_val + c_val)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Use single thread due to WAR dependencies
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s2244_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
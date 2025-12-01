import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be efficiently parallelized due to the sequential nature
    # of the packing operation where j is incremented based on conditions
    # We'll use a single thread block approach
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    j = -1
    
    for i in range(n_elements):
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Check condition
        if a_val > 0.0:
            j += 1
            # Load b[j] and store to a[i]
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    
    # Use single thread block since this is inherently sequential
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
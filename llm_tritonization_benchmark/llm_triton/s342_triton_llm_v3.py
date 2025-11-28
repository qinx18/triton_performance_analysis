import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be parallelized due to the sequential dependency on j
    # We need to process elements sequentially
    pid = tl.program_id(axis=0)
    
    # Only process if this is the first (and only) block
    if pid != 0:
        return
    
    j = -1
    
    # Process elements sequentially
    for i in range(0, n_elements):
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Check if a[i] > 0
        if a_val > 0.0:
            j += 1
            # Load b[j] and store to a[i]
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    
    # Use only one thread block since this must be sequential
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
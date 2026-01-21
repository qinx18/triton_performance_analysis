import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire array sequentially since j depends on previous iterations
    # We'll use a single thread block to maintain the sequential dependency
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize j counter
    j = -1
    
    # Process elements sequentially in chunks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load a values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n:
                break
            
            # Check if a[i] > 0
            a_val = tl.load(a_ptr + block_start + i)
            if a_val > 0.0:
                j += 1
                b_val = tl.load(b_ptr + j)
                tl.store(a_ptr + block_start + i, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Since this has sequential dependencies, we use only one block
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
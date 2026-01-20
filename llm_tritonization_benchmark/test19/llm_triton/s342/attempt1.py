import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize j for this block
    j = -1
    
    # Process elements sequentially within this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n:
            break
            
        # Load a[idx]
        a_val = tl.load(a_ptr + idx)
        
        # Check condition
        if a_val > 0.0:
            j = j + 1
            # Load b[j] and store to a[idx]
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + idx, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Since this algorithm has sequential dependencies, we need to process serially
    # We'll use a single thread block approach
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
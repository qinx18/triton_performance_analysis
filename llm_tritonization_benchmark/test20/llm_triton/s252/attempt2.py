import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This computation is inherently sequential due to t = s dependency
    # Process with single thread block
    pid = tl.program_id(0)
    
    if pid == 0:
        t = 0.0
        
        # Process elements one by one to maintain dependency
        for i in range(N):
            # Load single elements
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            
            # Compute s and a[i]
            s = b_val * c_val
            a_val = s + t
            
            # Store result
            tl.store(a_ptr + i, a_val)
            
            # Update t for next iteration
            t = s

def s252_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Use single thread block for sequential computation
    grid = (1,)
    
    s252_kernel[grid](
        a, b, c, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, LEN_1D, n1, n3, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the sequential loop
    num_iters = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Process iterations sequentially since k depends on previous iterations
    j = 1
    k = 0
    
    for iter_idx in range(num_iters):
        # Calculate current i value
        i = (n1 - 1) + iter_idx * n3
        
        # Update k
        k += j
        
        # Calculate indices for memory access
        b_idx = LEN_1D - k
        
        # Load values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + b_idx)
        
        # Update a[i]
        result = a_val + b_val
        tl.store(a_ptr + i, result)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    BLOCK_SIZE = 128
    
    # Launch kernel with single program instance since we need sequential execution
    grid = (1,)
    s122_kernel[grid](
        a, b, LEN_1D, n1, n3, BLOCK_SIZE
    )
    
    return a
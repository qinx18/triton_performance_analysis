import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Sequential computation cannot be parallelized due to k dependency
    # Process entire sequence in single thread block
    if tl.program_id(0) != 0:
        return
    
    j = 1
    k = 0
    
    # Calculate number of iterations
    num_iters = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    for iter_idx in range(num_iters):
        i = (n1 - 1) + iter_idx * n3
        if i >= LEN_1D:
            break
            
        k += j
        b_idx = LEN_1D - k
        
        if b_idx >= 0 and b_idx < LEN_1D:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + b_idx)
            tl.store(a_ptr + i, a_val + b_val)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch single block since computation is sequential
    grid = (1,)
    
    s122_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )
    
    return a
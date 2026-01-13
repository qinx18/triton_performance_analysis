import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate number of iterations
    num_iters = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get block ID and calculate which iteration this block handles
    block_id = tl.program_id(0)
    
    if block_id >= num_iters:
        return
    
    # Calculate the index i for this iteration
    i = (n1 - 1) + block_id * n3
    
    if i >= N:
        return
    
    # Calculate k for this iteration: k = j * (block_id + 1), where j = 1
    k = block_id + 1
    
    # Calculate indices
    b_idx = N - k
    
    if b_idx >= 0 and b_idx < N:
        # Load values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + b_idx)
        
        # Compute and store
        result = a_val + b_val
        tl.store(a_ptr + i, result)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate number of iterations
    if n1 - 1 >= N:
        return
    
    num_iters = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_iters <= 0:
        return
    
    BLOCK_SIZE = 1
    grid = (num_iters,)
    
    s122_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE)
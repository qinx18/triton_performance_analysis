import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, len_1d, BLOCK_SIZE: tl.constexpr):
    # Calculate number of iterations in the original loop
    start_idx = n1 - 1
    if start_idx >= len_1d:
        return
    
    num_iterations = (len_1d - start_idx + n3 - 1) // n3
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which iteration this block handles
    if pid >= num_iterations:
        return
    
    # Calculate the actual index in the original loop
    i = start_idx + pid * n3
    
    # Load and compute
    if i < len_1d:
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        result = a_val + b_val
        tl.store(a_ptr + i, result)

def s172_triton(a, b, n1, n3):
    len_1d = a.shape[0]
    
    # Calculate number of iterations
    start_idx = n1 - 1
    if start_idx >= len_1d:
        return
    
    num_iterations = (len_1d - start_idx + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    BLOCK_SIZE = 1
    grid = (num_iterations,)
    
    s172_kernel[grid](
        a, b, n1, n3, len_1d,
        BLOCK_SIZE=BLOCK_SIZE
    )
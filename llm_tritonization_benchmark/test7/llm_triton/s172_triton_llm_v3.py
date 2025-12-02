import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate number of iterations in the strided loop
    num_iterations = (n_elements - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate iteration index
    pid = tl.program_id(0)
    
    if pid >= num_iterations:
        return
    
    # Calculate the actual array index: i = (n1-1) + pid * n3
    i = (n1 - 1) + pid * n3
    
    # Load values
    a_val = tl.load(a_ptr + i)
    b_val = tl.load(b_ptr + i)
    
    # Compute and store result
    result = a_val + b_val
    tl.store(a_ptr + i, result)

def s172_triton(a, b, n1, n3):
    n_elements = a.shape[0]
    
    # Calculate number of iterations in the strided loop
    if n1 - 1 >= n_elements or n3 <= 0:
        return
    
    num_iterations = (n_elements - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    BLOCK_SIZE = 128
    grid = (num_iterations,)
    
    s172_kernel[grid](
        a, b, n1, n3, n_elements, BLOCK_SIZE
    )
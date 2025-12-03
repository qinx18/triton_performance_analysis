import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Each program handles one iteration of the original loop
    pid = tl.program_id(0)
    
    # Calculate the i value for this iteration
    i = n1 - 1 + pid * n3
    
    # Check if this iteration is within bounds
    if i >= LEN_1D:
        return
    
    # Calculate k = j * (pid + 1), where j = 1
    k = pid + 1
    
    # Calculate indices
    b_idx = LEN_1D - k
    
    # Load values
    a_val = tl.load(a_ptr + i)
    b_val = tl.load(b_ptr + b_idx)
    
    # Compute and store result
    result = a_val + b_val
    tl.store(a_ptr + i, result)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations
    if n1 - 1 >= LEN_1D or n3 <= 0:
        return  # No iterations
    
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (num_iterations,)
    
    s122_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )
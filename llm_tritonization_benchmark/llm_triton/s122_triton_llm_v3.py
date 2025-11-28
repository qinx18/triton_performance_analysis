import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which iteration this thread handles
    if pid >= num_iterations:
        return
    
    # Calculate i, j, k for this iteration
    i = n1 - 1 + pid * n3
    j = 1
    k = j * (pid + 1)  # k accumulates j for each iteration
    
    # Bounds check
    if i >= LEN_1D:
        return
    
    # Load values
    a_val = tl.load(a_ptr + i)
    b_idx = LEN_1D - k
    
    # Bounds check for b array access
    if b_idx >= 0 and b_idx < LEN_1D:
        b_val = tl.load(b_ptr + b_idx)
        result = a_val + b_val
        tl.store(a_ptr + i, result)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate the number of iterations
    num_iterations = max(0, (LEN_1D - (n1 - 1) + n3 - 1) // n3)
    
    if num_iterations == 0:
        return
    
    # Launch kernel with one thread per iteration
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )
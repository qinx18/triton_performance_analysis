import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate the iteration index for this block
    block_id = tl.program_id(0)
    
    # Calculate how many iterations this kernel will handle
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Each block handles one iteration
    if block_id >= num_iterations:
        return
    
    # Calculate i, j, k for this iteration
    i = (n1 - 1) + block_id * n3
    j = 1
    k = j * (block_id + 1)  # k accumulates j for each iteration
    
    # Bounds check
    if i >= LEN_1D:
        return
    
    # Calculate indices
    b_idx = LEN_1D - k
    
    # Bounds check for b array access
    if b_idx < 0 or b_idx >= LEN_1D:
        return
    
    # Load values
    a_val = tl.load(a_ptr + i)
    b_val = tl.load(b_ptr + b_idx)
    
    # Compute and store
    result = a_val + b_val
    tl.store(a_ptr + i, result)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations needed
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    # Launch kernel with one thread per iteration
    BLOCK_SIZE = 1
    grid = (num_iterations,)
    
    s122_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )
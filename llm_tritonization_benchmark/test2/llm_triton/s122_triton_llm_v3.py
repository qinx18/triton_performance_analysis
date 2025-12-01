import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate iteration index
    iter_idx = tl.program_id(0)
    
    # Calculate the actual i value for this iteration
    i = n1 - 1 + iter_idx * n3
    
    # Check if this iteration is valid
    if i >= LEN_1D:
        return
    
    # Calculate k = j * (iter_idx + 1), where j = 1
    k = iter_idx + 1
    
    # Calculate indices
    b_idx = LEN_1D - k
    
    # Check bounds
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
    
    # Calculate number of iterations
    num_iters = 0
    for i in range(n1-1, LEN_1D, n3):
        num_iters += 1
    
    if num_iters == 0:
        return
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (num_iters,)
    
    s122_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )
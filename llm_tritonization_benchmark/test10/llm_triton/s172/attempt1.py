import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate starting index for this program
    pid = tl.program_id(0)
    
    # Calculate the actual index using n1-1 + pid*n3
    idx = n1 - 1 + pid * n3
    
    # Check if index is within bounds
    if idx < LEN_1D:
        # Load values
        a_val = tl.load(a_ptr + idx)
        b_val = tl.load(b_ptr + idx)
        
        # Compute and store
        result = a_val + b_val
        tl.store(a_ptr + idx, result)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations needed
    if n3 <= 0:
        return
    
    start_idx = n1 - 1
    if start_idx >= LEN_1D:
        return
    
    # Calculate number of valid indices
    num_indices = (LEN_1D - start_idx + n3 - 1) // n3
    
    if num_indices <= 0:
        return
    
    # Launch kernel with one thread per valid index
    grid = (num_indices,)
    BLOCK_SIZE = 1
    
    s172_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )
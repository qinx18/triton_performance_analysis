import torch
import triton
import triton.language as tl

@triton.jit
def s317_kernel():
    q = 1.0
    for i in range(16000):  # LEN_1D/2 where LEN_1D is typically 32000
        q *= 0.99
    
    # Store result to global memory
    result_ptr = tl.load(tl.program_id(0))
    tl.store(result_ptr, q)

def s317_triton():
    # Create a tensor to store the result
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    # Launch kernel with single thread since this is a scalar reduction
    grid = (1,)
    s317_kernel[grid]()
    
    return result.item()
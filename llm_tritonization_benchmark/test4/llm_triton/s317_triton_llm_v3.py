import torch
import triton
import triton.language as tl

@triton.jit
def s317_kernel(result_ptr, BLOCK_SIZE: tl.constexpr, N: tl.constexpr):
    # Get program ID for this block
    pid = tl.program_id(0)
    
    # Initialize accumulator
    q = 1.0
    
    # Compute product reduction
    for i in range(N):
        q *= 0.99
    
    # Store result
    if pid == 0:
        tl.store(result_ptr, q)

def s317_triton():
    LEN_1D = 32000
    N = LEN_1D // 2
    
    # Allocate output tensor
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    # Launch kernel with single block since this is a scalar reduction
    BLOCK_SIZE = 128
    grid = (1,)
    
    s317_kernel[grid](
        result,
        BLOCK_SIZE=BLOCK_SIZE,
        N=N,
    )
    
    return result.item()
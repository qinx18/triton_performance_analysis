import triton
import triton.language as tl
import torch

@triton.jit
def s313_kernel(a_ptr, b_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_dot = 0.0
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        block_dot += tl.sum(a_vals * b_vals, axis=0)
    
    tl.store(result_ptr, block_dot)

def s313_triton(a, b):
    N = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block
    grid = (1,)
    s313_kernel[grid](
        a, b, result,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
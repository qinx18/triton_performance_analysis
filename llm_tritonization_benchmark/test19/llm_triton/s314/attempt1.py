import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element value
    first_val = tl.load(a_ptr)
    block_max = tl.full([BLOCK_SIZE], first_val, dtype=tl.float32)
    
    # Process array in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        block_max = tl.maximum(block_max, vals)
    
    # Reduce within block
    final_max = tl.max(block_max)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, final_max)

def s314_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch single block kernel
    grid = (1,)
    s314_kernel[grid](a, result, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()
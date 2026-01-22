import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    j = -1
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which elements are negative
        negative_mask = a_vals < 0.0
        
        # For each negative element in this block, update j
        for i in range(BLOCK_SIZE):
            if block_start + i < n:
                if negative_mask & (1 << i):
                    j = block_start + i
    
    # Store the final j value
    tl.store(result_ptr, j)

def s331_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for the result
    result = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single thread since we need sequential processing
    grid = (1,)
    s331_kernel[grid](a, result, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()
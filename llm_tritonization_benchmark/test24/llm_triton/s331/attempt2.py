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
        
        # For each element in this block, update j if negative
        for i in range(BLOCK_SIZE):
            idx = block_start + i
            if idx < n:
                # Extract single element to check
                single_mask = offsets == i
                element = tl.sum(tl.where(single_mask, a_vals, 0.0))
                if element < 0.0:
                    j = idx
    
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
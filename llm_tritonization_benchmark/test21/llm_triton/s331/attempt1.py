import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Find the last index where a[i] < 0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result to -1
    tl.store(result_ptr, -1)
    
    # Process array in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which elements are negative
        negative_mask = a_vals < 0.0
        valid_negative_mask = mask & negative_mask
        
        # Update result for each negative element (later indices will overwrite earlier ones)
        for i in range(BLOCK_SIZE):
            if i + block_start < N:
                if tl.load(a_ptr + (block_start + i)) < 0.0:
                    tl.store(result_ptr, block_start + i)

def s331_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    s331_kernel[grid](a, result, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()
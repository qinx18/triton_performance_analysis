import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Find the last index where a[i] < 0
    current_j = -1
    
    # Process array in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load block of data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Find negative values
        neg_mask = a_vals < 0.0
        
        # For each element in the block that is negative, update j
        for i in range(BLOCK_SIZE):
            if block_start + i < N:
                if neg_mask & (1 << i):
                    current_j = block_start + i
    
    # Store result
    if tl.program_id(0) == 0:
        tl.store(result_ptr, current_j)

def s331_triton(a):
    N = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Since this is a sequential search for the last occurrence,
    # we need to use a single thread block to maintain order
    BLOCK_SIZE = 256
    grid = (1,)
    
    s331_kernel[grid](a, result, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()
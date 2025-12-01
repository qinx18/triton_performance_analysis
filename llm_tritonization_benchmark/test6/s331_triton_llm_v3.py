import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find the last index where a[i] < 0, or -1 if none found
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result to -1
    last_negative_idx = -1
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which values are negative
        negative_mask = vals < 0.0
        
        # For each negative value in this block, update last_negative_idx
        for i in range(BLOCK_SIZE):
            if i + block_start < n_elements:
                if negative_mask[i]:
                    last_negative_idx = i + block_start
    
    # Only first thread writes the result
    if tl.program_id(0) == 0:
        tl.store(result_ptr, last_negative_idx)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    # Launch kernel with single block since we need sequential processing
    grid = (1,)
    s331_kernel[grid](
        a, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
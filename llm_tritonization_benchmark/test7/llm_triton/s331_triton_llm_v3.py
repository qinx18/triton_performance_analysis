import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result to -1
    j = -1
    
    # Process the array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from array a
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which values are negative
        negative_mask = vals < 0.0
        
        # For each negative value, update j to the corresponding index
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                if i < BLOCK_SIZE and negative_mask[i]:
                    j = block_start + i
    
    # Store the result
    tl.store(result_ptr, j)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for result
    result = torch.tensor(-1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single thread since we need sequential processing
    grid = (1,)
    s331_kernel[grid](a, result, n_elements, BLOCK_SIZE)
    
    return result.item()
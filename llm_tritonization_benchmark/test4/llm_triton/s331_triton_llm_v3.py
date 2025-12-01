import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize j to -1
    j = -1
    
    # Process all elements sequentially to maintain dependency
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load array elements
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check condition for each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                if a_vals[i] < 0.0:
                    j = block_start + i
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, j)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single block to maintain sequential dependency
    grid = (1,)
    s331_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
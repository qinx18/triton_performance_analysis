import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Product reduction - use single thread/block for sequential multiplication
    if tl.program_id(0) > 0:
        return
    
    prod = 1.0
    
    # Define offsets once at start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Multiply all values in block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                # Extract scalar value using tl.sum with specific mask
                single_mask = offsets == i
                val = tl.sum(tl.where(single_mask, vals, 0.0))
                prod *= val
    
    # Store result
    tl.store(result_ptr, prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block for reduction
    grid = (1,)
    s312_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
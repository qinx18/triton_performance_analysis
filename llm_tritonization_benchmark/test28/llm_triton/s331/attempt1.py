import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result to -1
    if tl.program_id(0) == 0:
        tl.store(result_ptr, -1)
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which elements are negative
        neg_mask = vals < 0.0
        combined_mask = mask & neg_mask
        
        # For each negative element, update result with its index
        for i in range(BLOCK_SIZE):
            if combined_mask[i]:
                idx = block_start + i
                tl.atomic_max(result_ptr, idx)

def s331_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s331_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
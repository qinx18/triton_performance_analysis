import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Check which elements are negative
    neg_mask = vals < 0.0
    combined_mask = mask & neg_mask
    
    # Find the highest index where element is negative
    indices = current_offsets
    neg_indices = tl.where(combined_mask, indices, -1)
    max_neg_idx = tl.max(neg_indices, axis=0)
    
    # Atomic update with maximum index found
    tl.atomic_max(result_ptr, max_neg_idx)

def s331_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor initialized to -1
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s331_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
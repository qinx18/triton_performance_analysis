import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    j = -1
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which elements are negative
        negative_mask = a_vals < 0.0
        
        # Update j for each negative element in this block
        neg_indices = tl.where(negative_mask, current_offsets, -1)
        
        # Find the maximum valid negative index in this block
        max_neg_idx = tl.max(tl.where(mask & negative_mask, neg_indices, -1))
        
        # Update j if we found a valid negative element
        j = tl.maximum(j, max_neg_idx)
    
    tl.store(result_ptr, j)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    grid = (1,)
    s331_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
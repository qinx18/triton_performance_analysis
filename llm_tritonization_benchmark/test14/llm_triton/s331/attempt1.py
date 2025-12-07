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
        neg_mask = a_vals < 0.0
        valid_neg_mask = mask & neg_mask
        
        # If any elements in this block are negative, update j
        if tl.sum(valid_neg_mask.to(tl.int32)) > 0:
            # Find the last negative element in this block
            for i in range(BLOCK_SIZE):
                if i < BLOCK_SIZE and (block_start + i) < n_elements:
                    if valid_neg_mask[i]:
                        j = block_start + i
    
    tl.store(result_ptr, j)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.tensor(-1, dtype=torch.int32, device=a.device)
    
    grid = (1,)
    s331_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
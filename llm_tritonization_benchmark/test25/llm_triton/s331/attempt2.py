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
        
        # Find the highest index with negative value in this block
        for i in range(BLOCK_SIZE):
            idx = block_start + i
            if idx < n_elements:
                element_mask = (offsets == i)
                is_negative = tl.sum(tl.where(element_mask & neg_mask, 1, 0))
                if is_negative > 0:
                    j = idx
    
    tl.store(result_ptr, j)

def s331_triton(a):
    N = a.shape[0]
    
    # Allocate result tensor
    result = torch.tensor(-1, dtype=torch.int32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s331_kernel[grid](
        a, result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
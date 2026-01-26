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
        
        # Update j for each negative element found
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                if neg_mask & (1 << i):  # Check if i-th element is negative
                    j = block_start + i
    
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
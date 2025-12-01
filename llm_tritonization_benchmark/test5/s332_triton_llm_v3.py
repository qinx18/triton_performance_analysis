import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find first value greater than threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    
    index = -2
    value = -1.0
    found = False
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        if found:
            break
            
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any value in this block is greater than threshold
        greater_mask = a_vals > t
        if tl.any(greater_mask):
            # Find the first index where condition is true
            for i in range(BLOCK_SIZE):
                local_offset = block_start + i
                if local_offset < n_elements:
                    local_mask = current_offsets == local_offset
                    local_val = tl.sum(tl.where(local_mask, a_vals, 0.0))
                    if local_val > t:
                        index = local_offset
                        value = local_val
                        found = True
                        break
    
    chksum = value + index
    tl.store(result_ptr, chksum)

def s332_triton(a, t_val):
    n_elements = a.shape[0]
    
    # Allocate result tensor
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s332_kernel[grid](
        a, t_val, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()
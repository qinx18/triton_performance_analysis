import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, threshold, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    found_index = -2
    found_value = -1.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        values = tl.load(a_ptr + current_offsets, mask=mask, other=threshold)
        
        # Check if any value in this block is greater than threshold
        condition = values > threshold
        has_match = tl.sum(condition.to(tl.int32)) > 0
        
        if has_match:
            # Find the first matching element in this block
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    offset = block_start + i
                    val = tl.load(a_ptr + offset)
                    if val > threshold:
                        found_index = offset
                        found_value = val
                        break
            break
    
    chksum = found_value + found_index
    tl.store(result_ptr, chksum)

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Create output tensor for result
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block to maintain sequential search
    
    s332_kernel[grid](
        a, result, 
        float(t), n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
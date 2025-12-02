import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find maximum value and its index using reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    # Process array in blocks to find global maximum
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in this block
        block_max = tl.max(vals)
        
        # Check if this block's max is greater than current global max
        if block_max > max_val:
            max_val = block_max
            # Find the index of the maximum value in this block
            max_mask = vals == block_max
            # Get the first index where maximum occurs
            indices = tl.where(max_mask, current_offsets, n_elements)
            max_idx = tl.min(indices)
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor to store [max_value, max_index]
    result = torch.zeros(2, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s315_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    max_idx = int(result[1].item())
    chksum = max_val + float(max_idx)
    
    return max_idx + max_val + 1
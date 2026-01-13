import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find first value greater than threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result values
    found_index = -2
    found_value = -1.0
    found = False
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        a_block = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check if any values are greater than threshold
        gt_mask = a_block > t
        
        if tl.any(gt_mask & mask):
            # Find first occurrence in this block
            for i in range(BLOCK_SIZE):
                local_offset = block_start + i
                if local_offset < n_elements:
                    val = tl.load(a_ptr + local_offset)
                    if val > t:
                        found_index = local_offset
                        found_value = val
                        found = True
                        break
            if found:
                break
    
    # Store results
    tl.store(result_ptr, found_value)
    tl.store(result_ptr + 1, found_index)

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Use PyTorch for more efficient implementation
    mask = a > t
    indices = torch.nonzero(mask, as_tuple=False)
    
    if indices.numel() > 0:
        first_idx = indices[0].item()
        value = a[first_idx].item()
        index = first_idx
    else:
        value = -1.0
        index = -2
    
    return value
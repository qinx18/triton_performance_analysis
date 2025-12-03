import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result with default values (index=-2, value=-1.0)
    index = -2
    value = -1.0
    found = False
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=t - 1.0)
        
        # Check which elements are greater than threshold
        gt_mask = a_vals > t
        gt_mask = gt_mask & mask
        
        # Check if any element in this block satisfies condition
        has_match = tl.sum(tl.where(gt_mask, 1, 0)) > 0
        
        if has_match:
            # Find first element in this block that satisfies condition
            for i in range(BLOCK_SIZE):
                global_idx = block_start + i
                if global_idx < n_elements:
                    val = tl.load(a_ptr + global_idx)
                    if val > t:
                        index = global_idx
                        value = val
                        found = True
                        break
            if found:
                break
    
    # Store results
    chksum = value + index
    tl.store(result_ptr, chksum)

def s332_triton(a, t):
    n_elements = a.numel()
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s332_kernel[grid](
        a, t, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()
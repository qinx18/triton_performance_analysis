import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find first value greater than threshold
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
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=t - 1.0)  # other < t
        
        # Check which elements are greater than threshold
        gt_mask = a_vals > t
        
        if tl.any(gt_mask):
            # Find first element in this block that satisfies condition
            for i in range(BLOCK_SIZE):
                local_offset = offsets[i]
                global_idx = block_start + local_offset
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
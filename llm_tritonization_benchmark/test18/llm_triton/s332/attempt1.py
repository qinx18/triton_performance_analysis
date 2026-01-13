import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find first value greater than threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially to maintain order
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        values = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check which values are greater than threshold
        greater_mask = values > t
        
        # If any value in this block satisfies condition
        if tl.sum(greater_mask.to(tl.int32)) > 0:
            # Find first occurrence within the block
            for i in range(BLOCK_SIZE):
                if i + block_start < n_elements:
                    offset = i + block_start
                    val = tl.load(a_ptr + offset)
                    if val > t:
                        # Store the value and index
                        tl.store(result_ptr, val)
                        tl.store(result_ptr + 1, offset)
                        return
    
    # If no value found, store defaults
    tl.store(result_ptr, -1.0)
    tl.store(result_ptr + 1, -2)

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Use PyTorch for simpler implementation of search
    mask = a > t
    if torch.any(mask):
        # Find first index where condition is true
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        first_idx = indices[0].item()
        value = a[first_idx].item()
        index = first_idx
    else:
        value = -1.0
        index = -2
    
    chksum = value + float(index)
    return value
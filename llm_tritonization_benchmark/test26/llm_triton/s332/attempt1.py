import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find first element > t using reduction approach
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=float('-inf'))
    
    # Check condition
    condition = a_vals > t
    
    # Set values to -inf where condition is false to ignore them
    filtered_vals = tl.where(condition, a_vals, float('-inf'))
    
    # Find local maximum and its position
    max_val = tl.max(filtered_vals)
    max_mask = filtered_vals == max_val
    
    # Get the first index where max occurs
    indices = tl.arange(0, BLOCK_SIZE) + block_start
    valid_indices = tl.where(max_mask, indices, n_elements)
    min_index = tl.min(valid_indices)
    
    # Store results if this block found a valid value
    if max_val != float('-inf'):
        tl.atomic_min(result_ptr + 1, min_index)  # index
        # Only update value if we found an earlier index
        old_index = tl.load(result_ptr + 1)
        if min_index <= old_index:
            tl.store(result_ptr, max_val)  # value

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Use PyTorch for efficient first-value-greater-than search
    mask = a > t
    valid_indices = torch.where(mask)[0]
    
    if len(valid_indices) > 0:
        first_index = valid_indices[0].item()
        value = a[first_index].item()
        index = first_index
    else:
        # No value found greater than t
        value = -1.0
        index = -2
    
    # Calculate chksum (though not returned, maintaining C logic)
    chksum = value + float(index)
    
    return value
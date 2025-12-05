import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program processes one block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load block of data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=float('-inf'))
    
    # Find elements greater than threshold
    greater_mask = a_vals > t
    
    # Check if any element in this block satisfies condition
    if tl.sum(greater_mask.to(tl.int32)) > 0:
        # Find first occurrence using cumulative sum
        indices = tl.arange(0, BLOCK_SIZE)
        global_indices = block_start + indices
        
        # Create mask for valid positions that satisfy condition
        valid_greater = greater_mask & mask
        
        if tl.sum(valid_greater.to(tl.int32)) > 0:
            # Use where to find first valid index
            first_valid_local = tl.minimum(indices + tl.where(valid_greater, 0, BLOCK_SIZE), BLOCK_SIZE - 1)
            min_local_idx = tl.minimum(tl.reduce(first_valid_local, 0, tl.minimum), BLOCK_SIZE - 1)
            
            # Check if this is actually a valid hit
            actual_mask = (indices == min_local_idx) & valid_greater
            if tl.sum(actual_mask.to(tl.int32)) > 0:
                found_index = tl.sum(tl.where(actual_mask, global_indices, 0))
                found_value = tl.sum(tl.where(actual_mask, a_vals, 0.0))
                
                # Store result atomically (only first thread that finds a value wins)
                tl.atomic_min(result_ptr, found_index)
                tl.atomic_min(result_ptr + 1, found_value)

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Initialize result tensor with default values
    result = torch.tensor([n_elements, -1.0], dtype=a.dtype, device=a.device)
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # First pass: find minimum index of elements > t
    s332_kernel[(num_blocks,)](
        a, result, t, n_elements, BLOCK_SIZE
    )
    
    index = int(result[0].item())
    
    # Check if we found a valid result
    if index < n_elements:
        value = a[index].item()
        final_index = index
    else:
        # No element found, use default values
        final_index = -2
        value = -1.0
    
    chksum = value + float(final_index)
    
    return chksum
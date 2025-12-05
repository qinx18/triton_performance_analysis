import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load block of data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=float('-inf'))
    
    # Find elements greater than threshold
    greater_mask = a_vals > t
    valid_greater = greater_mask & mask
    
    # Check if any element in this block satisfies condition
    has_match = tl.sum(valid_greater.to(tl.int32)) > 0
    
    if has_match:
        # Create a large value for positions that don't match
        indices_with_penalty = tl.where(valid_greater, block_start + offsets, n_elements)
        
        # Find the minimum index (first occurrence)
        min_global_idx = tl.min(indices_with_penalty)
        
        # Check if this minimum is actually valid
        if min_global_idx < n_elements:
            # Find which local position corresponds to this global index
            local_match = (block_start + offsets) == min_global_idx
            final_match = local_match & valid_greater
            
            if tl.sum(final_match.to(tl.int32)) > 0:
                found_value = tl.sum(tl.where(final_match, a_vals, 0.0))
                
                # Store result atomically
                tl.atomic_min(result_ptr, min_global_idx)
                tl.atomic_min(result_ptr + 1, found_value.to(tl.int32).to(tl.float32))

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Initialize result tensor with large values for atomic_min
    result = torch.tensor([n_elements, float('inf')], dtype=a.dtype, device=a.device)
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s332_kernel[(num_blocks,)](
        a, result, float(t), n_elements, BLOCK_SIZE
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
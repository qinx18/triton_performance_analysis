import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel to find first value greater than threshold
    Uses parallel reduction to find minimum index across blocks
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data with masking for out-of-bounds
    mask = offsets < n_elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find elements greater than threshold
    gt_mask = a_vals > t
    valid_mask = mask & gt_mask
    
    # Create indices for this block
    indices = offsets
    
    # Set invalid indices to large value for min reduction
    indices = tl.where(valid_mask, indices, n_elements)
    
    # Find minimum valid index in this block
    min_idx = tl.min(indices)
    
    # Store result: [min_index, corresponding_value, found_flag]
    # Use atomic min to find global minimum across blocks
    if min_idx < n_elements:
        # Found valid element in this block
        corresponding_val = tl.load(a_ptr + min_idx)
        # Atomic compare-and-swap for minimum index
        old_idx = tl.atomic_cas(result_ptr, n_elements, min_idx)
        if old_idx == n_elements or min_idx < old_idx:
            # We have the new minimum, update value too
            tl.store(result_ptr + 1, corresponding_val)
            tl.store(result_ptr + 2, 1.0)  # found flag

def s332_triton(a, t):
    """
    Triton implementation of TSVC s332 - first value greater than threshold
    Optimized with parallel search and atomic operations for global minimum
    """
    a = a.contiguous()
    n_elements = a.numel()
    
    # Result buffer: [index, value, found_flag]
    result = torch.full((3,), n_elements, dtype=torch.float32, device=a.device)
    result[1] = -1.0  # default value
    result[2] = 0.0   # found flag
    
    if n_elements == 0:
        index = torch.tensor(-2, device=a.device, dtype=torch.long)
        value = torch.tensor(-1.0, device=a.device, dtype=a.dtype)
        return a
    
    # Launch kernel with appropriate block size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s332_kernel[grid](
        a, result, t, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Extract results
    found = result[2].item() > 0
    if found:
        index = result[0].to(torch.long)
        value = result[1].to(a.dtype)
    else:
        index = torch.tensor(-2, device=a.device, dtype=torch.long)
        value = torch.tensor(-1.0, device=a.device, dtype=a.dtype)
    
    # Compute checksum (matching original implementation)
    chksum = value + index.to(a.dtype)
    
    return a
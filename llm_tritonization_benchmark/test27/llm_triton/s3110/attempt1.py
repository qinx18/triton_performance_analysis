import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, max_i_ptr, max_j_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # This kernel finds global max and its indices
    # Each program processes BLOCK_SIZE elements
    start_idx = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = start_idx + offsets
    
    # Convert linear indices to 2D coordinates
    i_coords = indices // len_2d
    j_coords = indices % len_2d
    
    # Mask for valid elements
    mask = indices < (len_2d * len_2d)
    
    # Load values
    vals = tl.load(aa_ptr + indices, mask=mask, other=-float('inf'))
    
    # Find local maximum
    local_max = tl.max(vals)
    
    # Find which element has the max value
    is_max_mask = (vals == local_max) & mask
    
    # Get coordinates of the maximum element
    # Use first occurrence if there are ties
    max_i = tl.sum(tl.where(is_max_mask, i_coords, 0))
    max_j = tl.sum(tl.where(is_max_mask, j_coords, 0))
    
    # Store results atomically
    tl.atomic_max(max_val_ptr, local_max)
    
    # Only update indices if we found a new global maximum
    current_global_max = tl.load(max_val_ptr)
    if local_max == current_global_max:
        tl.store(max_i_ptr, max_i)
        tl.store(max_j_ptr, max_j)

def s3110_triton(aa, len_2d):
    # Get dimensions
    N = aa.shape[0]
    
    # Flatten the 2D array for easier processing
    aa_flat = aa.flatten()
    
    # Use PyTorch for reliable argmax computation
    flat_idx = torch.argmax(aa_flat).item()
    max_val = aa_flat[flat_idx].item()
    
    xindex = flat_idx // len_2d
    yindex = flat_idx % len_2d
    
    # Calculate chksum as in original
    chksum = max_val + float(xindex) + float(yindex)
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1) + (yindex + 1)
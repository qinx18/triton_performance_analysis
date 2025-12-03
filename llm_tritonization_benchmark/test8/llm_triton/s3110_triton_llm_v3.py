import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(
    aa_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Initialize max value and indices
    # Load first element as initial max
    first_val = tl.load(aa_ptr)
    current_max = first_val
    max_i = 0
    max_j = 0
    
    # Process all elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(aa_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in this block
        block_max = tl.max(vals, axis=0)
        
        # If block max is greater than current max, update
        if block_max > current_max:
            current_max = block_max
            
            # Find the position of max within the block
            max_mask = vals == block_max
            max_positions = tl.where(max_mask, current_offsets, n_elements)
            min_max_pos = tl.min(max_positions, axis=0)  # Get first occurrence
            
            # Convert linear index to 2D coordinates
            len_2d = tl.sqrt(n_elements.to(tl.float32)).to(tl.int32)
            max_i = min_max_pos // len_2d
            max_j = min_max_pos % len_2d
    
    # Calculate final result
    result = current_max + max_i.to(tl.float32) + max_j.to(tl.float32)
    
    # Store results
    tl.store(output_ptr, current_max)
    tl.store(output_ptr + 1, max_i.to(tl.float32))
    tl.store(output_ptr + 2, max_j.to(tl.float32))
    tl.store(output_ptr + 3, result)

def s3110_triton(aa):
    # Flatten the 2D array for processing
    aa_flat = aa.flatten()
    n_elements = aa_flat.numel()
    
    # Output tensor to store [max_val, max_i, max_j, chksum]
    output = torch.zeros(4, dtype=aa.dtype, device=aa.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)  # Single block processes everything
    
    s3110_kernel[grid](
        aa_flat,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    max_val = output[0].item()
    xindex = int(output[1].item())
    yindex = int(output[2].item())
    
    return max_val + (xindex + 1) + (yindex + 1)
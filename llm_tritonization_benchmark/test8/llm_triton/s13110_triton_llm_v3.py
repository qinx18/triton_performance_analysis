import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_out_ptr, xindex_out_ptr, yindex_out_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Initialize values
    current_max = tl.load(aa_ptr)  # aa[0][0]
    current_xindex = 0
    current_yindex = 0
    
    # Process all elements in the 2D array
    for i in range(len_2d):
        for j_start in range(0, len_2d, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            j_offsets = j_start + offsets
            mask = j_offsets < len_2d
            
            # Load elements from current row
            row_ptr = aa_ptr + i * len_2d
            vals = tl.load(row_ptr + j_offsets, mask=mask, other=float('-inf'))
            
            # Find max in this block
            block_max = tl.max(vals)
            
            # Check if block max is greater than current max
            if block_max > current_max:
                # Find the exact position within the block
                max_mask = vals == block_max
                max_positions = tl.where(max_mask, j_offsets, len_2d)
                block_j = tl.min(max_positions)  # Get first occurrence
                
                current_max = block_max
                current_xindex = i
                current_yindex = block_j
    
    # Store results
    tl.store(max_out_ptr, current_max)
    tl.store(xindex_out_ptr, current_xindex)
    tl.store(yindex_out_ptr, current_yindex)

def s13110_triton(aa):
    len_2d = aa.shape[0]
    
    # Output tensors
    max_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    BLOCK_SIZE = 64
    
    # Launch kernel with single thread
    grid = (1,)
    s13110_kernel[grid](
        aa, max_out, xindex_out, yindex_out,
        len_2d, BLOCK_SIZE
    )
    
    max_val = max_out.item()
    xindex = xindex_out.item()
    yindex = yindex_out.item()
    
    return max_val + (xindex + 1) + (yindex + 1)
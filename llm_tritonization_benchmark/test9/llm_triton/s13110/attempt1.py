import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_out_ptr, xindex_out_ptr, yindex_out_ptr, chksum_out_ptr, 
                  LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each block processes one element to find global max
    block_id = tl.program_id(0)
    
    if block_id > 0:
        return
    
    # Initialize with aa[0][0]
    first_offset = tl.zeros([1], dtype=tl.int32)
    max_val = tl.load(aa_ptr + first_offset)
    max_val_scalar = tl.max(max_val)
    best_xindex = 0
    best_yindex = 0
    
    # Process all elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    total_size = LEN_2D * LEN_2D
    
    for block_start in range(0, total_size, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < total_size
        
        # Load block of values
        vals = tl.load(aa_ptr + current_offsets, mask=mask, other=-float('inf'))
        
        # Find max in this block
        block_max = tl.max(vals)
        
        # If this block has a new global max, find its position
        if block_max > max_val_scalar:
            # Find which element in the block is the max
            is_max_mask = vals == block_max
            
            # Get the first occurrence of max
            for k in range(BLOCK_SIZE):
                if block_start + k < total_size:
                    current_offset = block_start + k
                    if k < BLOCK_SIZE:
                        val_k = tl.load(aa_ptr + current_offset)
                        if val_k > max_val_scalar:
                            max_val_scalar = val_k
                            # Convert linear index to 2D coordinates
                            best_xindex = current_offset // LEN_2D
                            best_yindex = current_offset % LEN_2D
    
    # Store results
    tl.store(max_out_ptr, max_val_scalar)
    tl.store(xindex_out_ptr, best_xindex)
    tl.store(yindex_out_ptr, best_yindex)
    
    chksum = max_val_scalar + best_xindex + best_yindex
    tl.store(chksum_out_ptr, chksum)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Flatten the 2D array for easier processing
    aa_flat = aa.flatten()
    
    # Output tensors
    max_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    chksum_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s13110_kernel[grid](
        aa_flat,
        max_out,
        xindex_out, 
        yindex_out,
        chksum_out,
        LEN_2D,
        BLOCK_SIZE
    )
    
    return max_out.item(), xindex_out.item(), yindex_out.item()
import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value and its indices in a 2D array
    # Each block processes a portion of the array and finds local max
    
    block_id = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    
    # Calculate elements per block
    total_elements = LEN_2D * LEN_2D
    elements_per_block = tl.cdiv(total_elements, num_blocks)
    
    # Starting position for this block
    start_idx = block_id * elements_per_block
    
    # Initialize local max values
    local_max = float('-inf')
    local_xindex = 0
    local_yindex = 0
    
    # Process elements in this block
    for offset in range(0, elements_per_block, BLOCK_SIZE):
        # Calculate global indices
        indices = start_idx + offset + tl.arange(0, BLOCK_SIZE)
        mask = indices < total_elements
        
        # Convert linear indices to 2D coordinates
        i_coords = indices // LEN_2D
        j_coords = indices % LEN_2D
        
        # Load values from 2D array
        ptrs = aa_ptr + i_coords * LEN_2D + j_coords
        values = tl.load(ptrs, mask=mask, other=float('-inf'))
        
        # Find maximum in this chunk
        for k in range(BLOCK_SIZE):
            if mask[k]:
                if values[k] > local_max:
                    local_max = values[k]
                    local_xindex = i_coords[k]
                    local_yindex = j_coords[k]
    
    # Store results (will be reduced later on CPU)
    tl.store(max_ptr + block_id, local_max)
    tl.store(xindex_ptr + block_id, local_xindex)
    tl.store(yindex_ptr + block_id, local_yindex)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    device = aa.device
    
    # Number of blocks for parallel processing
    num_blocks = min(256, tl.cdiv(LEN_2D * LEN_2D, 256))
    BLOCK_SIZE = 256
    
    # Temporary arrays to store partial results from each block
    max_values = torch.full((num_blocks,), float('-inf'), device=device, dtype=aa.dtype)
    xindex_values = torch.zeros((num_blocks,), device=device, dtype=torch.int32)
    yindex_values = torch.zeros((num_blocks,), device=device, dtype=torch.int32)
    
    # Launch kernel
    grid = (num_blocks,)
    s13110_kernel[grid](
        aa, max_values, xindex_values, yindex_values,
        LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find global maximum from partial results
    max_idx = torch.argmax(max_values)
    max_val = max_values[max_idx]
    xindex = xindex_values[max_idx]
    yindex = yindex_values[max_idx]
    
    # Calculate checksum
    chksum = max_val + xindex.float() + yindex.float()
    
    return max_val + (xindex + 1) + (yindex + 1)
import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # This is a reduction operation to find max value and its indices in 2D array
    # Since it's a global reduction, we use a single thread block
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize with aa[0][0]
    current_max = tl.load(aa_ptr)
    current_xindex = 0
    current_yindex = 0
    
    # Create offset vectors once at start
    offsets_i = tl.arange(0, BLOCK_SIZE)
    offsets_j = tl.arange(0, BLOCK_SIZE)
    
    # Iterate through all elements
    for i_start in range(0, len_2d, BLOCK_SIZE):
        for j_start in range(0, len_2d, BLOCK_SIZE):
            # Compute current offsets
            i_offsets = i_start + offsets_i
            j_offsets = j_start + offsets_j
            
            # Create masks
            i_mask = i_offsets < len_2d
            j_mask = j_offsets < len_2d
            
            # Load block of data
            for i_local in range(BLOCK_SIZE):
                if i_start + i_local >= len_2d:
                    break
                    
                i_idx = i_start + i_local
                row_ptr = aa_ptr + i_idx * len_2d
                
                j_block_offsets = j_offsets
                mask = j_mask
                
                # Load a row block
                values = tl.load(row_ptr + j_block_offsets, mask=mask, other=float('-inf'))
                
                # Check each element in the block
                for j_local in range(BLOCK_SIZE):
                    if j_start + j_local >= len_2d:
                        break
                        
                    j_idx = j_start + j_local
                    val = tl.load(row_ptr + j_idx)
                    
                    if val > current_max:
                        current_max = val
                        current_xindex = i_idx
                        current_yindex = j_idx
    
    # Store results
    tl.store(max_val_ptr, current_max)
    tl.store(xindex_ptr, current_xindex)
    tl.store(yindex_ptr, current_yindex)

def s3110_triton(aa):
    len_2d = aa.shape[0]
    
    # Output tensors
    max_val = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    BLOCK_SIZE = 32
    
    # Launch kernel with single program
    grid = (1,)
    s3110_kernel[grid](
        aa, max_val, xindex, yindex, len_2d, BLOCK_SIZE
    )
    
    # Compute checksum
    chksum = max_val + xindex.float() + yindex.float()
    
    return max_val.item() + (xindex.item() + 1) + (yindex.item() + 1)
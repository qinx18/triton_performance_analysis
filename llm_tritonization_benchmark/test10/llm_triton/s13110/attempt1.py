import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, chksum_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize with aa[0][0]
    local_max = tl.load(aa_ptr)
    local_xindex = 0
    local_yindex = 0
    
    for i in range(LEN_2D):
        # Load entire row for valid j indices
        row_ptr = aa_ptr + i * LEN_2D + j_idx
        row_vals = tl.load(row_ptr, mask=j_mask, other=-float('inf'))
        
        # Find which elements are greater than current max
        greater_mask = (row_vals > local_max) & j_mask
        
        # Update max, xindex, yindex for any element that's greater
        if tl.sum(greater_mask.to(tl.int32)) > 0:
            # Find the maximum value among those greater than current max
            masked_vals = tl.where(greater_mask, row_vals, -float('inf'))
            new_max = tl.max(masked_vals)
            
            # Find the j index of this maximum value
            max_j_mask = (row_vals == new_max) & greater_mask
            j_indices = tl.where(max_j_mask, j_idx, LEN_2D)
            new_j = tl.min(j_indices)  # Get the smallest j index in case of ties
            
            # Update if we found a valid new maximum
            if new_max > local_max:
                local_max = new_max
                local_xindex = i
                local_yindex = new_j
    
    # Store results (only first thread writes the final result)
    if pid == 0:
        tl.store(max_val_ptr, local_max)
        tl.store(xindex_ptr, local_xindex)
        tl.store(yindex_ptr, local_yindex)
        chksum = local_max + local_xindex + local_yindex
        tl.store(chksum_ptr, chksum)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Output tensors
    max_val = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    chksum = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s13110_kernel[grid](
        aa, max_val, xindex, yindex, chksum,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return max_val.item() + (xindex.item() + 1) + (yindex.item() + 1)
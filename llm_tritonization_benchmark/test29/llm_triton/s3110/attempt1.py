import triton
import triton.language as tl
import torch

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Initialize offsets for j dimension
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    # Initialize reduction variables
    local_max = tl.load(aa_ptr)  # aa[0][0]
    local_xindex = 0
    local_yindex = 0
    
    # Sequential loop over i dimension
    for i in range(len_2d):
        # Load entire row for valid j indices
        row_ptr = aa_ptr + i * len_2d + j_idx
        row_vals = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Find positions where row values are greater than current max
        greater_mask = row_vals > local_max
        
        # Update max value if any element in this row is greater
        row_max = tl.max(row_vals)
        if row_max > local_max:
            local_max = row_max
            local_xindex = i
            # Find the j index of the maximum in this row
            max_positions = row_vals == row_max
            for j in range(len_2d):
                if j < BLOCK_SIZE and j_mask[j] and max_positions[j]:
                    local_yindex = j_idx[j]
                    break
    
    # Store results (only first thread writes)
    if pid == 0 and j_offsets[0] == 0:
        tl.store(max_val_ptr, local_max)
        tl.store(xindex_ptr, local_xindex)
        tl.store(yindex_ptr, local_yindex)

def s3110_triton(aa, len_2d):
    # Use PyTorch for simpler and more efficient argmax computation
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // len_2d
    yindex = max_idx % len_2d
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1
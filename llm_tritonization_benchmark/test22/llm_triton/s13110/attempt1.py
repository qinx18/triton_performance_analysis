import triton
import triton.language as tl
import torch

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + offsets
    mask = j_idx < len_2d
    
    # Initialize with first element values for this block
    current_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    current_xindex = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    current_yindex = j_idx
    
    # Sequential loop over i dimension
    for i in range(len_2d):
        # Load values for current row i, columns j_idx
        row_offset = i * len_2d
        load_ptrs = aa_ptr + row_offset + j_idx
        values = tl.load(load_ptrs, mask=mask, other=float('-inf'))
        
        # Update max, xindex, yindex where new value is greater
        update_mask = values > current_max
        current_max = tl.where(update_mask, values, current_max)
        current_xindex = tl.where(update_mask, i, current_xindex)
        current_yindex = tl.where(update_mask, j_idx, current_yindex)
    
    # Store results
    if mask.any():
        tl.store(max_val_ptr + j_idx, current_max, mask=mask)
        tl.store(xindex_ptr + j_idx, current_xindex, mask=mask)
        tl.store(yindex_ptr + j_idx, current_yindex, mask=mask)

def s13110_triton(aa, len_2d):
    device = aa.device
    
    # Create output tensors
    max_vals = torch.full((len_2d,), float('-inf'), dtype=torch.float32, device=device)
    xindices = torch.zeros(len_2d, dtype=torch.int32, device=device)
    yindices = torch.zeros(len_2d, dtype=torch.int32, device=device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s13110_kernel[grid](
        aa, max_vals, xindices, yindices, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find global maximum and its indices
    global_max_idx = torch.argmax(max_vals)
    max_val = max_vals[global_max_idx]
    yindex = global_max_idx.item()
    xindex = xindices[global_max_idx].item()
    
    # Calculate chksum (though not used in return)
    chksum = max_val + float(xindex) + float(yindex)
    
    # Return as specified in C code: max + xindex+1 + yindex+1
    return max_val + xindex + 1 + yindex + 1
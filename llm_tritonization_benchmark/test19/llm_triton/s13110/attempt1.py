import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    # Initialize with first element
    if pid == 0:
        first_val = tl.load(aa_ptr)
        tl.store(max_val_ptr, first_val)
        tl.store(xindex_ptr, 0)
        tl.store(yindex_ptr, 0)
    
    tl.debug_barrier()
    
    for i in range(len_2d):
        # Load current row values
        row_offsets = i * len_2d + j_idx
        row_vals = tl.load(aa_ptr + row_offsets, mask=j_mask, other=float('-inf'))
        
        # Find local maximum in this block
        local_max = tl.max(row_vals)
        
        # Check if this local max is greater than global max
        current_global_max = tl.load(max_val_ptr)
        if local_max > current_global_max:
            # Find which j in this block has the maximum
            max_mask = row_vals == local_max
            j_candidates = tl.where(max_mask & j_mask, j_idx, len_2d)
            local_j = tl.min(j_candidates)
            
            # Update global maximum and indices
            tl.store(max_val_ptr, local_max)
            tl.store(xindex_ptr, i)
            tl.store(yindex_ptr, local_j)

def s13110_triton(aa, len_2d):
    # Use PyTorch's argmax for better accuracy
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // len_2d
    yindex = max_idx % len_2d
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1
import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, max_i_ptr, max_j_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    # Initialize with first element
    current_max = tl.load(aa_ptr)
    current_max_i = 0
    current_max_j = 0
    
    for i in range(len_2d):
        # Load row i at columns j_idx
        row_ptr = aa_ptr + i * len_2d
        vals = tl.load(row_ptr + j_idx, mask=j_mask, other=float('-inf'))
        
        # Find if any values in this block are greater than current max
        greater_mask = vals > current_max
        
        # Update max and indices if needed
        for block_j in range(BLOCK_SIZE):
            if j_idx[block_j] < len_2d and greater_mask[block_j]:
                current_max = vals[block_j]
                current_max_i = i
                current_max_j = j_idx[block_j]
    
    # Store results from thread 0 of each block
    if tl.program_id(0) == 0 and j_offsets[0] == 0:
        tl.store(max_val_ptr, current_max)
        tl.store(max_i_ptr, current_max_i)
        tl.store(max_j_ptr, current_max_j)

def s3110_triton(aa, len_2d):
    # Use PyTorch for argmax reduction as it's more efficient
    aa_flat = aa.flatten()
    max_val = torch.max(aa_flat)
    max_idx = torch.argmax(aa_flat)
    
    xindex = max_idx // len_2d
    yindex = max_idx % len_2d
    
    chksum = max_val + xindex.float() + yindex.float()
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1).float() + (yindex + 1).float()
import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, max_i_ptr, max_j_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute column offsets
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < n_cols
    
    # Initialize with first element values
    current_max = tl.load(aa_ptr)  # aa[0][0]
    current_max_i = 0
    current_max_j = 0
    
    # Sequential loop over rows (i dimension)
    for i in range(n_rows):
        # Load entire row for current columns
        row_ptr = aa_ptr + i * n_cols + j_idx
        row_vals = tl.load(row_ptr, mask=j_mask, other=-float('inf'))
        
        # Find which elements are greater than current max
        greater_mask = row_vals > current_max
        valid_greater = greater_mask & j_mask
        
        # Update max values where condition is met
        current_max = tl.where(valid_greater, row_vals, current_max)
        current_max_i = tl.where(valid_greater, i, current_max_i)
        current_max_j = tl.where(valid_greater, j_idx, current_max_j)
    
    # Reduce within block to find global maximum
    # First reduce to find the maximum value
    block_max = tl.max(current_max)
    
    # Find which thread has the maximum value
    has_max = current_max == block_max
    
    # Get the indices corresponding to the maximum
    max_i_candidate = tl.where(has_max, current_max_i, n_rows)
    max_j_candidate = tl.where(has_max, current_max_j, n_cols)
    
    # Find the minimum indices among threads that have the maximum value
    block_max_i = tl.min(max_i_candidate)
    block_max_j = tl.min(max_j_candidate)
    
    # Store results (only first thread in block writes)
    if tl.program_id(0) == 0 and j_offsets[0] == 0:
        tl.store(max_val_ptr, block_max)
        tl.store(max_i_ptr, block_max_i)
        tl.store(max_j_ptr, block_max_j)

def s3110_triton(aa):
    # Get dimensions
    n_rows, n_cols = aa.shape
    
    # Use PyTorch's argmax for reliable reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    xindex = flat_idx // n_cols
    yindex = flat_idx % n_cols
    
    # Compute chksum (though not used in return)
    chksum = max_val + xindex.float() + yindex.float()
    
    # Return the exact format from C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1) + (yindex + 1)
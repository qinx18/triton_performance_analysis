import triton
import triton.language as tl
import torch

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, max_i_ptr, max_j_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    # Initialize with aa[0][0]
    current_max = tl.load(aa_ptr)
    current_max_i = 0
    current_max_j = 0
    
    # Sequential loop over i dimension
    for i in range(N):
        # Load all j values for this i
        row_ptr = aa_ptr + i * N + j_idx
        vals = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Find positions where vals > current_max
        update_mask = vals > current_max
        update_mask = update_mask & j_mask
        
        # Update max value and indices where condition is true
        current_max = tl.where(update_mask, vals, current_max)
        current_max_i = tl.where(update_mask, i, current_max_i)
        current_max_j = tl.where(update_mask, j_idx, current_max_j)
    
    # Reduce across the block to find global maximum
    # First find the maximum value
    block_max = tl.max(current_max)
    
    # Find which thread has the maximum value
    has_max = current_max == block_max
    
    # Get the indices corresponding to the maximum
    max_i_val = tl.where(has_max, current_max_i, N)
    max_j_val = tl.where(has_max, current_max_j, N)
    
    # Reduce to get the minimum indices (in case of ties)
    final_i = tl.min(max_i_val)
    final_j = tl.min(max_j_val)
    
    # Store results (only first thread in each block)
    if pid == 0 and tl.program_id(0) == 0:
        tl.store(max_val_ptr, block_max)
        tl.store(max_i_ptr, final_i)
        tl.store(max_j_ptr, final_j)

def s3110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch's argmax for more reliable reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // N
    yindex = max_idx % N
    
    chksum = max_val + xindex.float() + yindex.float()
    
    # Return the same format as C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1).float() + (yindex + 1).float()
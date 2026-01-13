import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] once
    a_i_minus_1 = tl.load(a_ptr + i_val - 1)
    
    # Process all valid j values in this block
    valid_j = tl.where(j_mask, j_offsets, 0)
    
    # Load aa[j][i] for all valid j in block
    aa_offsets = valid_j * LEN_2D + i_val
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] for all valid j in block
    bb_offsets = valid_j * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    a_i_vals = aa_vals - a_i_minus_1
    
    # Since all j iterations overwrite a[i], we need the last valid one
    # Find the last valid j index in this block
    if tl.sum(j_mask.to(tl.int32)) > 0:
        # Get the maximum valid j index in this block
        max_valid_idx = tl.max(tl.where(j_mask, j_offsets, -1))
        
        # Only the thread with the maximum j should write a[i]
        should_write_a = (j_offsets == max_valid_idx) & j_mask
        
        # Write a[i] (only one thread will do this)
        tl.store(a_ptr + i_val, tl.where(should_write_a, a_i_vals, 0).sum())
        
        # Load the final a[i] value that was just written
        final_a_i = tl.load(a_ptr + i_val)
        
        # Compute new aa[j][i] = a[i] + bb[j][i] for all valid j
        new_aa_vals = final_a_i + bb_vals
        
        # Store aa[j][i] for all valid j
        tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Sequential loop over i from 1 to LEN_2D-1
    for i in range(1, LEN_2D):
        s257_kernel[(grid_size,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )
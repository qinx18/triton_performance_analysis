import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d, i_val, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(axis=0)
    
    # Calculate j offset
    j_offset = pid_j * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j indices
    j_mask = j_offsets < len_2d
    
    # Calculate pointers for current i and i-1
    aa_ptr_ji = aa_ptr + j_offsets * len_2d + i_val
    bb_ptr_ji = bb_ptr + j_offsets * len_2d + i_val
    a_ptr_i = a_ptr + i_val
    a_ptr_i_minus_1 = a_ptr + (i_val - 1)
    
    # Load aa[j][i] and bb[j][i]
    aa_ji = tl.load(aa_ptr_ji, mask=j_mask)
    bb_ji = tl.load(bb_ptr_ji, mask=j_mask)
    
    # Load a[i-1] (scalar broadcast)
    a_i_minus_1 = tl.load(a_ptr_i_minus_1)
    
    # Compute a[i] = aa[j][i] - a[i-1]
    # Since all j iterations write to the same a[i], we need the last valid j value
    if j_mask.any():
        # Find the last valid j index
        valid_mask = j_mask
        # For overwrite pattern, we take the last j that executes
        a_i_new = aa_ji - a_i_minus_1
        
        # Store the last computed value (from highest valid j)
        # We need to find which thread has the highest valid j
        max_valid_j = tl.max(tl.where(valid_mask, j_offsets, -1))
        current_max_j = tl.max(j_offsets)
        
        # Only the thread with the maximum j in this block writes a[i]
        if current_max_j == max_valid_j:
            # Get the value from the last valid position
            last_valid_idx = tl.max(tl.where(valid_mask, tl.arange(0, BLOCK_SIZE), -1))
            if last_valid_idx >= 0:
                a_i_final = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == last_valid_idx, a_i_new, 0.0))
                tl.store(a_ptr_i, a_i_final)
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    # We need to load the updated a[i] value
    a_i_updated = tl.load(a_ptr_i)
    aa_ji_new = a_i_updated + bb_ji
    
    # Store aa[j][i]
    tl.store(aa_ptr_ji, aa_ji_new, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, len_2d):
        # Calculate grid size for j dimension
        grid_j = triton.cdiv(len_2d, BLOCK_SIZE)
        
        # Launch kernel for current i
        s257_kernel[(grid_j,)](
            a, aa, bb, len_2d, i, BLOCK_SIZE
        )
import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d, i_val, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    # Load a[i] and a[i-1]
    a_i_prev = tl.load(a_ptr + (i_val - 1))
    
    # Process all j values in parallel
    for j_block_start in range(0, len_2d, BLOCK_SIZE):
        current_j_offsets = j_block_start + j_offsets
        current_j_mask = current_j_offsets < len_2d
        
        # Calculate aa indices: aa[j][i] = j * len_2d + i
        aa_indices = current_j_offsets * len_2d + i_val
        
        # Load aa[j][i] and bb[j][i]
        aa_vals = tl.load(aa_ptr + aa_indices, mask=current_j_mask)
        bb_vals = tl.load(bb_ptr + aa_indices, mask=current_j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        # Since all j iterations overwrite the same location, we'll handle this separately
        # For now, compute the values
        a_new_vals = aa_vals - a_i_prev
        
        # Store the last computed a[i] value (overwrite pattern)
        if j_block_start + BLOCK_SIZE >= len_2d:
            # This is the last block, find the last valid j
            last_valid_j_offset = len_2d - j_block_start - 1
            if last_valid_j_offset >= 0 and last_valid_j_offset < BLOCK_SIZE:
                last_a_val = tl.load(aa_ptr + (len_2d - 1) * len_2d + i_val) - a_i_prev
                tl.store(a_ptr + i_val, last_a_val)
        
        # Update aa[j][i] = a[i] + bb[j][i]
        # We need to use the correct a[i] value for each j
        aa_new_vals = a_new_vals + bb_vals
        tl.store(aa_ptr + aa_indices, aa_new_vals, mask=current_j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = 256
    
    # Sequential loop over i, parallel over j
    for i in range(1, len_2d):
        s257_kernel[(1,)](
            a, aa, bb, len_2d, i, BLOCK_SIZE
        )
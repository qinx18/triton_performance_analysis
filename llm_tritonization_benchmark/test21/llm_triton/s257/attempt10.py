import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar value)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # For all j values in this block
    for j_block_start in range(0, LEN_2D, BLOCK_SIZE):
        current_j_offsets = j_block_start + j_offsets
        current_j_mask = current_j_offsets < LEN_2D
        
        # Calculate 2D indices for aa[j][i] and bb[j][i]
        aa_indices = current_j_offsets * LEN_2D + i
        bb_indices = current_j_offsets * LEN_2D + i
        
        # Load values
        aa_vals = tl.load(aa_ptr + aa_indices, mask=current_j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=current_j_mask, other=0.0)
        
        # Compute a[i] = aa[j][i] - a[i-1] (for all j in this block)
        a_vals = aa_vals - a_prev
        
        # Store the last computed a[i] value (overwrite pattern)
        if tl.sum(current_j_mask.to(tl.int32)) > 0:
            # Find the highest valid j index
            valid_j_indices = tl.where(current_j_mask, current_j_offsets, -1)
            max_valid_j = tl.max(valid_j_indices)
            
            # Get the corresponding a[i] value for the highest j
            max_j_offset = max_valid_j - j_block_start
            max_j_mask = j_offsets == max_j_offset
            final_a_val = tl.sum(tl.where(max_j_mask, a_vals, 0.0))
            
            # Store a[i] if this is a valid index
            if max_valid_j >= 0:
                tl.store(a_ptr + i, final_a_val)
        
        # Update aa[j][i] = a[i] + bb[j][i] for all j in this block
        aa_new_vals = a_vals + bb_vals
        tl.store(aa_ptr + aa_indices, aa_new_vals, mask=current_j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
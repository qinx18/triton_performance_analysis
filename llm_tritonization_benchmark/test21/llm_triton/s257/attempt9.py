import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate 2D indices for aa[j][i] and bb[j][i]
    aa_indices = j_offsets * LEN_2D + i
    bb_indices = j_offsets * LEN_2D + i
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Compute a[i] values for each j
    a_vals = aa_vals - a_prev
    
    # The last valid j determines the final a[i] value
    valid_mask = j_mask
    if tl.sum(valid_mask.to(tl.int32)) > 0:
        # Find the last valid j
        valid_j_indices = tl.where(valid_mask, j_offsets, -1)
        max_valid_j = tl.max(valid_j_indices)
        
        # Get a[i] value from the last j
        last_j_aa_idx = max_valid_j * LEN_2D + i
        last_aa_val = tl.load(aa_ptr + last_j_aa_idx)
        final_a_val = last_aa_val - a_prev
        tl.store(a_ptr + i, final_a_val)
        
        # Update aa[j][i] for all j using their respective a[i] values
        aa_new_vals = a_vals + bb_vals
        tl.store(aa_ptr + aa_indices, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
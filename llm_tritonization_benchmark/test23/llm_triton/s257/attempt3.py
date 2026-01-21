import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    j_offset = pid * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar)
    a_prev = tl.load(a_ptr + (i_val - 1))
    
    # Load aa[j][i] values
    aa_ptrs = aa_ptr + j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptrs, mask=j_mask)
    
    # Load bb[j][i] values
    bb_ptrs = bb_ptr + j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptrs, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    a_new_vals = aa_vals - a_prev
    
    # Store the final a[i] value (from last j iteration that executes)
    # Only the last block that has valid elements should store
    if tl.sum(j_mask.to(tl.int32)) > 0:
        # Get the highest valid j index in this block
        valid_indices = tl.where(j_mask, j_offsets, -1)
        max_valid_j = tl.max(valid_indices)
        
        # If this block contains the globally last j (LEN_2D - 1), store a[i]
        if max_valid_j == LEN_2D - 1:
            # Find local index of the last j
            last_j_local = LEN_2D - 1 - j_offset
            a_final = a_new_vals[last_j_local]
            tl.store(a_ptr + i_val, a_final)
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    aa_new_vals = a_new_vals + bb_vals
    
    # Store aa[j][i] values
    tl.store(aa_ptrs, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential over i (1 to LEN_2D-1)
    for i in range(1, LEN_2D):
        # Process all j values in parallel for this i
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
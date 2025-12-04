import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    block_start = tl.program_id(0) * BLOCK_SIZE
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = block_start + j_offsets
    j_mask = j_indices < LEN_2D
    
    # Calculate array indices
    # a[i] and a[i-1] indices
    a_i_idx = i_val
    a_i_minus_1_idx = i_val - 1
    
    # aa[j][i] and bb[j][i] indices (row-major: j * LEN_2D + i)
    aa_ji_offsets = j_indices * LEN_2D + i_val
    bb_ji_offsets = j_indices * LEN_2D + i_val
    
    # Load values
    a_i_minus_1 = tl.load(a_copy_ptr + a_i_minus_1_idx)  # scalar load from copy
    aa_ji_vals = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask)
    bb_ji_vals = tl.load(bb_ptr + bb_ji_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    # Since all j's produce different values for a[i], we need to handle this carefully
    # The C code overwrites a[i] for each j, so we take the last valid j value
    a_i_vals = aa_ji_vals - a_i_minus_1
    
    # Find the last valid j index to determine final a[i] value
    valid_mask = j_mask
    if tl.sum(valid_mask.to(tl.int32)) > 0:
        # Get the last valid a[i] value (highest j index in this block)
        last_valid_idx = tl.max(tl.where(valid_mask, j_indices, -1))
        final_a_i = tl.max(tl.where(valid_mask & (j_indices == last_valid_idx), a_i_vals, -1e9))
        
        # Store a[i] (scalar store)
        if block_start + BLOCK_SIZE - 1 >= LEN_2D - 1:  # Only last block writes a[i]
            tl.store(a_ptr + a_i_idx, final_a_i)
    
    # Compute aa[j][i] = a[i] + bb[j][i] using the computed a[i] values
    aa_ji_new = a_i_vals + bb_ji_vals
    
    # Store aa[j][i]
    tl.store(aa_ptr + aa_ji_offsets, aa_ji_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy of array a
    a_copy = a.clone()
    
    # Sequential loop over i dimension
    for i in range(1, LEN_2D):
        # Parallel execution over j dimension
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        
        s257_kernel[grid](
            a, a_copy, aa, bb, i, LEN_2D, BLOCK_SIZE
        )
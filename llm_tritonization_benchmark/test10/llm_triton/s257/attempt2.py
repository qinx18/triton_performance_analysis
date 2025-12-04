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
    # aa[j][i] and bb[j][i] indices (row-major: j * LEN_2D + i)
    aa_ji_offsets = j_indices * LEN_2D + i_val
    bb_ji_offsets = j_indices * LEN_2D + i_val
    
    # Load values
    a_i_minus_1 = tl.load(a_copy_ptr + i_val - 1)  # scalar load from copy
    aa_ji_vals = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask)
    bb_ji_vals = tl.load(bb_ptr + bb_ji_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for the last j
    # Since the C code overwrites a[i] for each j, we need the value from the last j
    a_i_vals = aa_ji_vals - a_i_minus_1
    
    # Store a[i] using the last valid j's computation
    # We need to find which thread handles the last valid j index
    valid_js = tl.where(j_mask, j_indices, -1)
    max_valid_j = tl.max(valid_js)
    
    # Only store a[i] once using the computation from the highest j in this block
    if tl.sum((j_indices == max_valid_j) & j_mask) > 0:
        final_a_i = tl.sum(tl.where((j_indices == max_valid_j) & j_mask, a_i_vals, 0.0))
        if tl.program_id(0) == (LEN_2D - 1) // BLOCK_SIZE:  # Only last block writes
            tl.store(a_ptr + i_val, final_a_i)
    
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
        # For each i, we need to compute a[i] from the LAST j iteration
        # First pass: compute all a[i] values and find the one from max j
        temp_a_vals = torch.zeros(LEN_2D, dtype=a.dtype, device=a.device)
        
        for j in range(LEN_2D):
            a_val = aa[j, i] - a_copy[i-1]
            temp_a_vals[j] = a_val
            aa[j, i] = a_val + bb[j, i]
        
        # Store the final a[i] value (from last j)
        a[i] = temp_a_vals[LEN_2D - 1]
        
        # Update the copy for next iteration
        a_copy[i] = a[i]
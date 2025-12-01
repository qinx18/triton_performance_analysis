import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(
    a_ptr,
    a_copy_ptr,
    aa_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_J: tl.constexpr,
):
    # Get block IDs
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    # Calculate starting indices
    start_i = block_i * BLOCK_SIZE_I + 1  # i starts from 1
    start_j = block_j * BLOCK_SIZE_J
    
    # Create ranges for this block
    i_range = start_i + tl.arange(0, BLOCK_SIZE_I)
    j_range = start_j + tl.arange(0, BLOCK_SIZE_J)
    
    # Create masks
    i_mask = i_range < LEN_2D
    j_mask = j_range < LEN_2D
    
    # For each i in this block
    for i_offset in range(BLOCK_SIZE_I):
        i_idx = start_i + i_offset
        if i_idx >= LEN_2D:
            break
            
        # Load a[i-1] from copy (read-only)
        a_prev = tl.load(a_copy_ptr + (i_idx - 1))
        
        # Process all j values for this i
        j_offsets = start_j + tl.arange(0, BLOCK_SIZE_J)
        j_mask_current = j_offsets < LEN_2D
        
        # Calculate 2D indices for aa and bb arrays
        aa_offsets = j_offsets * LEN_2D + i_idx
        bb_offsets = j_offsets * LEN_2D + i_idx
        
        # Load aa[j][i] and bb[j][i]
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask_current)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask_current)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_new = aa_vals - a_prev
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_vals
        
        # Store results
        tl.store(a_ptr + i_idx, a_new[0])  # Store first computed a[i] value
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask_current)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of array 'a' to handle WAR dependencies
    a_copy = a.clone()
    
    # Define block sizes
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 32
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(LEN_2D - 1, BLOCK_SIZE_I)  # -1 because i starts from 1
    grid_j = triton.cdiv(LEN_2D, BLOCK_SIZE_J)
    
    # Launch kernel
    s257_kernel[(grid_i, grid_j)](
        a,
        a_copy,
        aa,
        bb,
        LEN_2D,
        BLOCK_SIZE_I,
        BLOCK_SIZE_J,
    )
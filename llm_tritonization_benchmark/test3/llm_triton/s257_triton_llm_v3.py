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
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for i dimension (starting from 1)
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # Process all j values for this i
    j_start = 0
    num_j = LEN_2D
    
    for j_block_start in range(0, num_j, BLOCK_SIZE):
        j_offsets = j_block_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < LEN_2D
        
        # Calculate array offsets
        a_i_offset = i
        a_i_minus_1_offset = i - 1
        aa_offsets = j_offsets * LEN_2D + i
        bb_offsets = j_offsets * LEN_2D + i
        
        # Load values
        a_i_minus_1 = tl.load(a_copy_ptr + a_i_minus_1_offset)
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_new = aa_vals - a_i_minus_1
        
        # Store a[i] (broadcast to all valid j positions, but we only need one value)
        if j_block_start == 0:  # Only store once
            tl.store(a_ptr + a_i_offset, a_new[0])
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of array a
    a_copy = a.clone()
    
    # Launch kernel with grid size (LEN_2D - 1) for i from 1 to LEN_2D-1
    BLOCK_SIZE = 64
    grid = (LEN_2D - 1,)
    
    s257_kernel[grid](
        a,
        a_copy,
        aa,
        bb,
        LEN_2D,
        BLOCK_SIZE,
    )
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
    
    # Load a[i-1] (scalar value)
    a_i_minus_1 = tl.load(a_ptr + (i_val - 1))
    
    # Load aa[j][i] values for all j in this block
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] values for all j in this block  
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_ji = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for each j
    # The last j will overwrite previous values
    a_i_new = aa_ji - a_i_minus_1
    
    # For the overwrite pattern, we need the value from the highest valid j
    # Find the maximum valid j index in this block
    valid_j_indices = tl.where(j_mask, j_offsets, -1)
    max_valid_j = tl.max(valid_j_indices)
    
    # Get the a[i] value corresponding to the maximum j
    is_max_j = j_offsets == max_valid_j
    a_i_final = tl.sum(tl.where(is_max_j & j_mask, a_i_new, 0.0))
    
    # Store a[i] - only one thread per block should store
    if tl.sum(is_max_j & j_mask) > 0:
        tl.store(a_ptr + i_val, a_i_final)
    
    # Compute aa[j][i] = a[i] + bb[j][i] using the new a[i] value for each j
    new_aa_ji = a_i_new + bb_ji
    
    # Store updated aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa_ji, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    # Sequential loop over i dimension
    for i in range(1, LEN_2D):
        # Calculate grid size for j dimension
        grid_j = triton.cdiv(LEN_2D, BLOCK_SIZE)
        
        # Launch kernel for current i
        s257_kernel[(grid_j,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )
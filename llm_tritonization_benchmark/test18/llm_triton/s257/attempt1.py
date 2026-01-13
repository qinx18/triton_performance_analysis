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
    
    # Load a[i] and a[i-1]
    a_i = tl.load(a_ptr + i_val)
    a_i_minus_1 = tl.load(a_ptr + i_val - 1)
    
    # Load aa[j][i] values for all j in this block
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] values for all j in this block
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_ji = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    # Since this overwrites the same location, we need the final result
    # The last j iteration wins, so we find the maximum valid j
    valid_j_mask = j_mask
    if tl.sum(valid_j_mask.to(tl.int32)) > 0:
        # Find the last valid j in this block
        last_valid_idx = tl.max(tl.where(valid_j_mask, j_offsets, -1))
        if last_valid_idx >= 0:
            # Get the aa value for the last valid j
            last_aa_offset = last_valid_idx * LEN_2D + i_val
            last_aa_val = tl.load(aa_ptr + last_aa_offset)
            # Update a[i] with the final value
            new_a_i = last_aa_val - a_i_minus_1
            tl.store(a_ptr + i_val, new_a_i)
            a_i = new_a_i
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    new_aa_ji = a_i + bb_ji
    tl.store(aa_ptr + aa_offsets, new_aa_ji, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Sequential loop over i from 1 to LEN_2D-1
    for i in range(1, LEN_2D):
        s257_kernel[(grid_size,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )
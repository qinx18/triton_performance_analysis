import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Read a[i-1]
    a_i_minus_1 = tl.load(a_ptr + (i_val - 1))
    
    # Process all j values in parallel
    for j_block_start in range(0, LEN_2D, BLOCK_SIZE):
        current_j_offsets = j_block_start + j_offsets
        current_j_mask = current_j_offsets < LEN_2D
        
        # Load aa[j][i] and bb[j][i]
        aa_offsets = current_j_offsets * LEN_2D + i_val
        bb_offsets = current_j_offsets * LEN_2D + i_val
        
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=current_j_mask)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=current_j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_i_val = aa_vals - a_i_minus_1
        
        # Store a[i] for each valid j (last j iteration wins due to overwrite)
        valid_mask = current_j_mask
        for k in range(BLOCK_SIZE):
            if k + j_block_start < LEN_2D:
                single_a_val = tl.load(aa_ptr + (k + j_block_start) * LEN_2D + i_val) - a_i_minus_1
                tl.store(a_ptr + i_val, single_a_val)
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        # Need to reload a[i] after it's been stored
        a_i_final = tl.load(a_ptr + i_val)
        new_aa_vals = a_i_final + bb_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=current_j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(min(LEN_2D, 1024))
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        s257_kernel[(1,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )
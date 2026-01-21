import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[i-1] (scalar broadcast)
    a_i_minus_1 = tl.load(a_ptr + (i - 1))
    
    # Process all j values - last one wins for a[i]
    for block_start in range(0, LEN_2D, BLOCK_SIZE):
        current_j_offsets = block_start + j_offsets
        current_mask = current_j_offsets < LEN_2D
        
        # Load aa[j][i] for current block of j values
        aa_ji_offsets = current_j_offsets * LEN_2D + i
        aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=current_mask)
        
        # Load bb[j][i] for current block of j values
        bb_ji_offsets = current_j_offsets * LEN_2D + i
        bb_ji = tl.load(bb_ptr + bb_ji_offsets, mask=current_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1] for all j (last one wins)
        a_i_values = aa_ji - a_i_minus_1
        
        # Store each a[i] value (last valid one wins)
        valid_indices = tl.where(current_mask, current_j_offsets, LEN_2D)
        valid_mask = valid_indices < LEN_2D
        
        # For each element in the block, if it's valid, store a[i]
        tl.store(a_ptr + i, tl.where(valid_mask, a_i_values, 0.0))
        
        # Now load the final a[i] value to compute aa[j][i]
        final_a_i = tl.load(a_ptr + i)
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        new_aa_ji = final_a_i + bb_ji
        
        # Store updated aa[j][i] values
        tl.store(aa_ptr + aa_ji_offsets, new_aa_ji, mask=current_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
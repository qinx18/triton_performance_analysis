import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[i-1] (scalar)
    a_i_minus_1 = tl.load(a_ptr + (i - 1))
    
    # Process all j values in blocks
    final_a_i = a_i_minus_1
    
    for block_start in range(0, LEN_2D, BLOCK_SIZE):
        current_j_offsets = block_start + j_offsets
        current_mask = current_j_offsets < LEN_2D
        
        # Load aa[j][i] and bb[j][i] for current block
        aa_ji_offsets = current_j_offsets * LEN_2D + i
        bb_ji_offsets = current_j_offsets * LEN_2D + i
        
        aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=current_mask, other=0.0)
        bb_ji = tl.load(bb_ptr + bb_ji_offsets, mask=current_mask, other=0.0)
        
        # For each j in this block, compute a[i] = aa[j][i] - a[i-1]
        # Each j overwrites a[i], so we need the last valid j
        for j_local in range(BLOCK_SIZE):
            j_global = block_start + j_local
            if j_global < LEN_2D:
                # Compute a[i] = aa[j][i] - a[i-1]
                aa_val = tl.load(aa_ptr + j_global * LEN_2D + i)
                final_a_i = aa_val - a_i_minus_1
                
                # Compute aa[j][i] = a[i] + bb[j][i]
                bb_val = tl.load(bb_ptr + j_global * LEN_2D + i)
                new_aa_val = final_a_i + bb_val
                
                # Store updated aa[j][i]
                tl.store(aa_ptr + j_global * LEN_2D + i, new_aa_val)
    
    # Store final a[i]
    tl.store(a_ptr + i, final_a_i)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
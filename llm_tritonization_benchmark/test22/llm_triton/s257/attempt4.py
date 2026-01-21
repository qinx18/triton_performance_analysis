import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[i-1] (scalar value)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Process all j values in blocks
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        current_j = j_start + j_offsets
        j_mask = current_j < LEN_2D
        
        # Load aa[j][i] for current block of j values
        aa_offsets = current_j * LEN_2D + i
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        
        # Load bb[j][i] for current block of j values  
        bb_offsets = current_j * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1] for each j
        a_vals = aa_vals - a_prev
        
        # Compute aa[j][i] = a[i] + bb[j][i] for each j
        new_aa_vals = a_vals + bb_vals
        
        # Store updated aa[j][i] values
        tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)
    
    # Final a[i] value is from the last j iteration (j = LEN_2D - 1)
    last_j = LEN_2D - 1
    aa_last = tl.load(aa_ptr + last_j * LEN_2D + i)
    bb_last = tl.load(bb_ptr + last_j * LEN_2D + i)
    a_final = aa_last - bb_last
    tl.store(a_ptr + i, a_final)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process i values sequentially from 1 to LEN_2D-1
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](
            a, aa, bb, i,
            LEN_2D=LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )
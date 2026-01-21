import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar value)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Process all j values
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        current_j = j_start + j_offsets
        current_mask = current_j < LEN_2D
        
        # Load aa[j][i] for current block of j values
        aa_offsets = current_j * LEN_2D + i
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=current_mask)
        
        # Load bb[j][i] for current block of j values
        bb_offsets = current_j * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=current_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1] for each j
        a_new = aa_vals - a_prev
        
        # Compute aa[j][i] = a[i] + bb[j][i] for each j
        new_aa_vals = a_new + bb_vals
        
        # Store updated aa[j][i] values
        tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=current_mask)
        
        # Store a[i] (last valid value will win)
        for idx in range(BLOCK_SIZE):
            if j_start + idx < LEN_2D:
                final_a = tl.load(aa_ptr + (j_start + idx) * LEN_2D + i) - a_prev
                tl.store(a_ptr + i, final_a)

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
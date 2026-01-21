import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar value)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] for all j values
    aa_offsets = j_offsets * LEN_2D + i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] for all j values
    bb_offsets = j_offsets * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # For each j: a[i] = aa[j][i] - a[i-1], then aa[j][i] = a[i] + bb[j][i]
    a_vals = aa_vals - a_prev
    new_aa_vals = a_vals + bb_vals
    
    # Store updated aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)
    
    # The final a[i] value is from the last j iteration (j = LEN_2D - 1)
    final_a_val = tl.load(aa_ptr + (LEN_2D - 1) * LEN_2D + i) - a_prev
    tl.store(a_ptr + i, final_a_val)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Process i values sequentially from 1 to LEN_2D-1
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](
            a, aa, bb, i,
            LEN_2D=LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )
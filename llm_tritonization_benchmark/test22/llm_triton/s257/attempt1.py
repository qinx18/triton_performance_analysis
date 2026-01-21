import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    i = tl.program_id(0) + 1
    
    # Load a[i-1] (scalar value)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] for all valid j values
    aa_offsets = j_offsets * LEN_2D + i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j values
    # Since this overwrites the same location, the last valid j will win
    a_vals = aa_vals - a_prev
    
    # Store the final a[i] value (from the last j iteration)
    final_j = LEN_2D - 1
    final_aa_offset = final_j * LEN_2D + i
    final_aa_val = tl.load(aa_ptr + final_aa_offset)
    final_a_val = final_aa_val - a_prev
    tl.store(a_ptr + i, final_a_val)
    
    # Load bb[j][i] for all valid j values
    bb_offsets = j_offsets * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j values
    new_aa_vals = a_vals + bb_vals
    
    # Store updated aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Process i values sequentially from 1 to LEN_2D-1
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](
            a, aa, bb,
            LEN_2D=LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )
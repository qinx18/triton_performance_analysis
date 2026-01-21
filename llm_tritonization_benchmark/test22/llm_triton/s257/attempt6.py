import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i] and a[i-1]
    a_i = tl.load(a_ptr + i_val)
    a_i_minus_1 = tl.load(a_ptr + i_val - 1)
    
    # Load aa[j][i] and bb[j][i] for all j
    aa_offsets = j_offsets * LEN_2D + i_val
    bb_offsets = j_offsets * LEN_2D + i_val
    
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for each j (last j wins)
    new_a_vals = aa_vals - a_i_minus_1
    
    # Store the last computed value to a[i]
    if LEN_2D > 0:
        final_a_val = tl.load(aa_ptr + (LEN_2D - 1) * LEN_2D + i_val) - a_i_minus_1
        tl.store(a_ptr + i_val, final_a_val)
    
    # Compute aa[j][i] = a[i] + bb[j][i] using the updated a[i]
    updated_a_i = tl.load(a_ptr + i_val)
    new_aa_vals = updated_a_i + bb_vals
    
    # Store aa[j][i] for all j
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](
            a, aa, bb,
            i, LEN_2D, BLOCK_SIZE
        )
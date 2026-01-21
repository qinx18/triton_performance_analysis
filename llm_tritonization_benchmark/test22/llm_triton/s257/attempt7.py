import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1]
    a_i_minus_1 = tl.load(a_ptr + i_val - 1)
    
    # For each j, compute a[i] = aa[j][i] - a[i-1] and aa[j][i] = a[i] + bb[j][i]
    # Process all j values to find the final a[i] value (last j iteration wins)
    aa_offsets = j_offsets * LEN_2D + i_val
    bb_offsets = j_offsets * LEN_2D + i_val
    
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute new a[i] values for each j
    new_a_vals = aa_vals - a_i_minus_1
    
    # Store each a[i] and then update aa[j][i]
    for j in range(LEN_2D):
        if j < LEN_2D:
            # a[i] = aa[j][i] - a[i-1]
            aa_j_i = tl.load(aa_ptr + j * LEN_2D + i_val)
            new_a_i = aa_j_i - a_i_minus_1
            tl.store(a_ptr + i_val, new_a_i)
            
            # aa[j][i] = a[i] + bb[j][i]
            bb_j_i = tl.load(bb_ptr + j * LEN_2D + i_val)
            new_aa_j_i = new_a_i + bb_j_i
            tl.store(aa_ptr + j * LEN_2D + i_val, new_aa_j_i)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](
            a, aa, bb,
            i, LEN_2D, BLOCK_SIZE
        )
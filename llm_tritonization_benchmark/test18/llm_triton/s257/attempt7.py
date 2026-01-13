import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1]
    a_i_minus_1 = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] for all j
    aa_ji_ptrs = aa_ptr + j_offsets * LEN_2D + i
    aa_ji_vals = tl.load(aa_ji_ptrs, mask=j_mask)
    
    # Load bb[j][i] for all j
    bb_ji_ptrs = bb_ptr + j_offsets * LEN_2D + i
    bb_ji_vals = tl.load(bb_ji_ptrs, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    # Since all j values write to same a[i], use the last valid j
    new_a_i_vals = aa_ji_vals - a_i_minus_1
    
    # Find the final value using reduction to get last valid element
    final_a_i = tl.load(aa_ptr + (LEN_2D - 1) * LEN_2D + i) - a_i_minus_1
    
    # Store final a[i]
    tl.store(a_ptr + i, final_a_i)
    
    # Compute and store aa[j][i] = a[i] + bb[j][i] for all j
    new_aa_ji = final_a_i + bb_ji_vals
    tl.store(aa_ji_ptrs, new_aa_ji, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Launch kernel for each i from 1 to LEN_2D-1 sequentially
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )
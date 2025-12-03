import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate base addresses for row i
    aa_base = aa_ptr + i_val * LEN_2D
    bb_base = bb_ptr + i_val * LEN_2D
    
    # Load aa[i][:] and bb[i][:]
    aa_vals = tl.load(aa_base + j_offsets, mask=j_mask)
    bb_vals = tl.load(bb_base + j_offsets, mask=j_mask)
    
    # Load cc[:][i] (transposed access)
    cc_vals = tl.load(cc_ptr + j_offsets * LEN_2D + i_val, mask=j_mask)
    
    # Compute aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
    result = aa_vals * cc_vals + bb_vals
    
    # Store back to aa[i][:]
    tl.store(aa_base + j_offsets, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Sequential loop over i, parallel over j
    for i in range(LEN_2D):
        s1115_kernel[(1,)](
            aa, bb, cc, i, LEN_2D, BLOCK_SIZE
        )
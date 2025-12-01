import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    i_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Vectorized j indices
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate memory offsets for aa[i_val][j] and bb[i_val][j]
    aa_offsets = i_val * LEN_2D + j_offsets
    bb_offsets = i_val * LEN_2D + j_offsets
    
    # Calculate memory offsets for cc[j][i_val] (transposed access)
    cc_offsets = j_offsets * LEN_2D + i_val
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask)
    
    # Compute: aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
    result = aa_vals * cc_vals + bb_vals
    
    # Store result
    tl.store(aa_ptr + aa_offsets, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Sequential loop over i, parallel over j
    for i in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1115_kernel[grid](
            aa, bb, cc,
            LEN_2D=LEN_2D,
            i_val=i,
            BLOCK_SIZE=BLOCK_SIZE,
        )
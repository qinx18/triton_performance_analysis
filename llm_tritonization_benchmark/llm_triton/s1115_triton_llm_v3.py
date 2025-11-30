import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Calculate memory offsets
    aa_offsets = i_val * LEN_2D + j_offsets
    bb_offsets = i_val * LEN_2D + j_offsets
    cc_offsets = j_offsets * LEN_2D + i_val
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    # Compute: aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
    result = aa_vals * cc_vals + bb_vals
    
    # Store result
    tl.store(aa_ptr + aa_offsets, result, mask=mask)

def s1115_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential over i, parallel over j
    for i_val in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1115_kernel[grid](aa, bb, cc, i_val, LEN_2D, BLOCK_SIZE)
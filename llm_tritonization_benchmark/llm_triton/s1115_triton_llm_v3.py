import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    for j in range(0, LEN_2D, BLOCK_SIZE):
        j_offsets = j + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < LEN_2D
        
        # Calculate flat indices
        aa_indices = i * LEN_2D + j_offsets
        bb_indices = i * LEN_2D + j_offsets
        cc_indices = j_offsets * LEN_2D + i
        
        # Load data
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask)
        cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
        
        # Compute: aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result
        tl.store(aa_ptr + aa_indices, result, mask=mask)

def s1115_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Flatten 2D arrays for kernel access
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    cc_flat = cc.flatten()
    
    grid = (LEN_2D,)
    
    s1115_kernel[grid](
        aa_flat, bb_flat, cc_flat,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa
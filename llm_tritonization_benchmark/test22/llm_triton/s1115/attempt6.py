import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(LEN_2D):
        # Load aa[i][j]
        aa_offsets = i * LEN_2D + j_idx
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask, other=0.0)
        
        # Load cc[j][i]
        cc_offsets = j_idx * LEN_2D + i
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
        
        # Load bb[i][j]
        bb_offsets = i * LEN_2D + j_idx
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Compute: aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result
        tl.store(aa_ptr + aa_offsets, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc,
        LEN_2D, BLOCK_SIZE
    )
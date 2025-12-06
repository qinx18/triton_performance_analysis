import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = pid * BLOCK_SIZE + offsets
    mask = i_offsets < LEN_2D
    
    # Calculate diagonal indices: i*LEN_2D + i
    diag_indices = i_offsets * LEN_2D + i_offsets
    
    # Load diagonal elements
    bb_vals = tl.load(bb_ptr + diag_indices, mask=mask)
    cc_vals = tl.load(cc_ptr + diag_indices, mask=mask)
    aa_vals = tl.load(aa_ptr + diag_indices, mask=mask)
    
    # Compute: aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(aa_ptr + diag_indices, result, mask=mask)

def s2101_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    s2101_kernel[(num_blocks,)](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
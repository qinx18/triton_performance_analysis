import triton
import triton.language as tl
import torch

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < LEN_2D
    
    # Calculate diagonal offsets: i * LEN_2D + i for aa[i][i], bb[i][i], cc[i][i]
    diag_offsets = indices * LEN_2D + indices
    
    # Load diagonal elements
    aa_vals = tl.load(aa_ptr + diag_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + diag_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + diag_offsets, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(aa_ptr + diag_offsets, result, mask=mask)

def s2101_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Ensure tensors are contiguous
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa
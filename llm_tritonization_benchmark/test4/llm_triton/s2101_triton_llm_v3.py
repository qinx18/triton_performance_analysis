import triton
import triton.language as tl
import torch

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < len_2d
    
    # Calculate diagonal addresses: aa[i][i] = aa_ptr + i * len_2d + i
    diag_offsets = indices * len_2d + indices
    
    # Load diagonal elements
    bb_diag = tl.load(bb_ptr + diag_offsets, mask=mask)
    cc_diag = tl.load(cc_ptr + diag_offsets, mask=mask)
    aa_diag = tl.load(aa_ptr + diag_offsets, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_diag + bb_diag * cc_diag
    
    # Store result
    tl.store(aa_ptr + diag_offsets, result, mask=mask)

def s2101_triton(aa, bb, cc):
    len_2d = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc,
        len_2d=len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
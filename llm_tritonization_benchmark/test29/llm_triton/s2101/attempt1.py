import triton
import triton.language as tl
import torch

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Each program handles one diagonal element
    pid = tl.program_id(0)
    
    if pid >= len_2d:
        return
    
    # Calculate diagonal index (i, i)
    diag_idx = pid * len_2d + pid
    
    # Load diagonal elements
    bb_val = tl.load(bb_ptr + diag_idx)
    cc_val = tl.load(cc_ptr + diag_idx)
    aa_val = tl.load(aa_ptr + diag_idx)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_val + bb_val * cc_val
    
    # Store result
    tl.store(aa_ptr + diag_idx, result)

def s2101_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
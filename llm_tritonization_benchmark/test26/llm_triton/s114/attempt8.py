import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import numpy as np
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(n):
        j_start = pid * BLOCK_SIZE
        j_idx = j_start + j_offsets
        
        j_mask = j_idx < i
        j_mask = j_mask & (j_idx < n)
        
        aa_read_idx = j_idx * n + i
        aa_write_idx = i * n + j_idx
        bb_idx = i * n + j_idx
        
        aa_vals = tl.load(aa_ptr + aa_read_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals
        
        tl.store(aa_ptr + aa_write_idx, result, mask=j_mask)

def s114_triton(aa, bb):
    n = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, n, BLOCK_SIZE)
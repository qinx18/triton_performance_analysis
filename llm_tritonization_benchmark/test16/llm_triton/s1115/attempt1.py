import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < n
    
    for i in range(n):
        aa_row_ptr = aa_ptr + i * n
        bb_row_ptr = bb_ptr + i * n
        
        aa_vals = tl.load(aa_row_ptr + j_idx, mask=j_mask)
        bb_vals = tl.load(bb_row_ptr + j_idx, mask=j_mask)
        
        cc_col_ptrs = cc_ptr + j_idx * n + i
        cc_vals = tl.load(cc_col_ptrs, mask=j_mask)
        
        result = aa_vals * cc_vals + bb_vals
        
        tl.store(aa_row_ptr + j_idx, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc, n, BLOCK_SIZE
    )
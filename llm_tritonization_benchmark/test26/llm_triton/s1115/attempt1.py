import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < n_cols
    
    for i in range(n_rows):
        aa_ptrs = aa_ptr + i * n_cols + j_idx
        bb_ptrs = bb_ptr + i * n_cols + j_idx
        cc_ptrs = cc_ptr + j_idx * n_cols + i
        
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        
        result = aa_vals * cc_vals + bb_vals
        
        tl.store(aa_ptrs, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    n_rows = aa.shape[0]
    n_cols = aa.shape[1]
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_cols, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc,
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
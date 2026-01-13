import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, n_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(n_size):
        current_i_idx = i_idx
        mask = (current_i_idx < n_size) & (current_i_idx >= j)
        
        bb_ptrs = bb_ptr + current_i_idx * n_size + j
        cc_ptrs = cc_ptr + current_i_idx * n_size + j
        aa_ptrs = aa_ptr + current_i_idx * n_size + j
        
        bb_vals = tl.load(bb_ptrs, mask=mask)
        cc_vals = tl.load(cc_ptrs, mask=mask)
        
        result = bb_vals + cc_vals
        
        tl.store(aa_ptrs, result, mask=mask)

def s1232_triton(aa, bb, cc):
    n_size = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_size, BLOCK_SIZE),)
    
    s1232_kernel[grid](aa, bb, cc, n_size, BLOCK_SIZE)
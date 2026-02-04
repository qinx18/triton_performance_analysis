import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr):
    j_idx = tl.program_id(0)
    
    for diag in range(2, 2 * N):
        i = diag - j_idx
        
        valid = (i >= 1) & (i < N) & (j_idx >= 1) & (j_idx < N)
        
        if valid:
            read_offset = (i - 1) * N + (j_idx - 1)
            write_offset = i * N + j_idx
            bb_offset = i * N + j_idx
            
            aa_val = tl.load(aa_ptr + read_offset)
            bb_val = tl.load(bb_ptr + bb_offset)
            
            result = aa_val + bb_val
            
            tl.store(aa_ptr + write_offset, result)

def s119_triton(aa, bb, len_2d):
    N = aa.shape[0]
    
    grid = (N,)
    s119_kernel[grid](aa, bb, N)
    
    return aa
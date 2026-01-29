import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        aa_curr_ptrs = aa_ptr + i * N + j_idx
        aa_prev_ptrs = aa_ptr + (i - 1) * N + j_idx
        bb_ptrs = bb_ptr + i * N + j_idx
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=j_mask)
        bb_vals = tl.load(bb_ptrs, mask=j_mask)
        
        result = aa_prev_vals + bb_vals
        
        tl.store(aa_curr_ptrs, result, mask=j_mask)

def s1119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(N))
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1119_kernel[grid](
        aa, bb, N, BLOCK_SIZE
    )
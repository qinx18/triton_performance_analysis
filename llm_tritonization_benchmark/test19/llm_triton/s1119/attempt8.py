import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_start = pid * BLOCK_SIZE
    j_idx = j_start + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        aa_curr_offsets = i * N + j_idx
        aa_prev_offsets = (i - 1) * N + j_idx
        bb_curr_offsets = i * N + j_idx
        
        aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=j_mask, other=0.0)
        bb_curr = tl.load(bb_ptr + bb_curr_offsets, mask=j_mask, other=0.0)
        
        result = aa_prev + bb_curr
        tl.store(aa_ptr + aa_curr_offsets, result, mask=j_mask)

def s1119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(N))
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, N, BLOCK_SIZE)
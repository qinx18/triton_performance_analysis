import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        # Load aa[i-1][j]
        aa_prev_ptrs = aa_ptr + (i-1) * N + j_idx
        aa_prev = tl.load(aa_prev_ptrs, mask=j_mask, other=0.0)
        
        # Load bb[i][j]
        bb_ptrs = bb_ptr + i * N + j_idx
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
        result = aa_prev + bb_vals
        
        # Store aa[i][j]
        aa_ptrs = aa_ptr + i * N + j_idx
        tl.store(aa_ptrs, result, mask=j_mask)

def s1119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1119_kernel[grid](aa.data_ptr(), bb.data_ptr(), N, BLOCK_SIZE=BLOCK_SIZE)
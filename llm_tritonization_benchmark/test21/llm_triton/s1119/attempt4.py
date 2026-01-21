import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_base = pid * BLOCK_SIZE
    j_idx = j_base + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        # Current row indices
        curr_row_base = i * N
        prev_row_base = (i - 1) * N
        
        # Compute full indices
        curr_aa_idx = curr_row_base + j_idx
        prev_aa_idx = prev_row_base + j_idx
        curr_bb_idx = curr_row_base + j_idx
        
        # Load values
        aa_prev = tl.load(aa_ptr + prev_aa_idx, mask=j_mask, other=0.0)
        bb_curr = tl.load(bb_ptr + curr_bb_idx, mask=j_mask, other=0.0)
        
        # Compute and store
        result = aa_prev + bb_curr
        tl.store(aa_ptr + curr_aa_idx, result, mask=j_mask)

def s1119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)
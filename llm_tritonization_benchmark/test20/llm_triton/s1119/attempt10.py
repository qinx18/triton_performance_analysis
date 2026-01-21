import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        # Load aa[i-1][j] for all j values in this block
        aa_prev_idx = (i - 1) * N + j_idx
        aa_prev = tl.load(aa_ptr + aa_prev_idx, mask=j_mask, other=0.0)
        
        # Load bb[i][j] for all j values in this block  
        bb_idx = i * N + j_idx
        bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
        result = aa_prev + bb_vals
        
        # Store back to aa[i][j]
        aa_curr_idx = i * N + j_idx
        tl.store(aa_ptr + aa_curr_idx, result, mask=j_mask)

def s1119_triton(aa, bb):
    N = aa.shape[0]
    
    BLOCK_SIZE = min(triton.next_power_of_2(N), 1024)
    
    # Ensure contiguous memory layout
    aa_flat = aa.view(-1).contiguous()
    bb_flat = bb.view(-1).contiguous()
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1119_kernel[grid](aa_flat, bb_flat, N=N, BLOCK_SIZE=BLOCK_SIZE)
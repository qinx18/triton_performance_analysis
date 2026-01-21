import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < N
    
    for i in range(1, N):
        # Check if i <= j for each thread
        valid_mask = j_mask & (i <= j_idx)
        
        # Calculate memory addresses for current and previous positions
        curr_addr = j_idx * N + i
        prev_addr = j_idx * N + (i - 1)
        
        # Load current aa[j][i] and bb[j][i]
        aa_curr = tl.load(aa_ptr + curr_addr, mask=valid_mask, other=0.0)
        bb_curr = tl.load(bb_ptr + curr_addr, mask=valid_mask, other=0.0)
        
        # Load aa[j][i-1]
        aa_prev = tl.load(aa_ptr + prev_addr, mask=valid_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i]
        result = aa_prev * aa_prev + bb_curr
        
        # Store result
        tl.store(aa_ptr + curr_addr, result, mask=valid_mask)

def s232_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    s232_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa
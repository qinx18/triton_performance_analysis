import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        i_mask = j_idx >= i
        valid_mask = j_mask & i_mask
        
        # Load previous values aa[j][i-1]
        prev_ptrs = aa_ptr + j_idx * N + (i - 1)
        prev_vals = tl.load(prev_ptrs, mask=valid_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + j_idx * N + i
        bb_vals = tl.load(bb_ptrs, mask=valid_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        new_vals = prev_vals * prev_vals + bb_vals
        
        # Store result
        curr_ptrs = aa_ptr + j_idx * N + i
        tl.store(curr_ptrs, new_vals, mask=valid_mask)

def s232_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s232_kernel[grid](
        aa, bb, N, BLOCK_SIZE
    )
    
    return aa
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
        # Each j processes its own triangular region: i <= j
        # So j must be >= i to participate
        valid_j_mask = j_mask & (j_idx >= i)
        
        # Load aa[j][i-1] for valid j indices
        prev_ptrs = aa_ptr + j_idx * N + (i - 1)
        prev_vals = tl.load(prev_ptrs, mask=valid_j_mask, other=0.0)
        
        # Load bb[j][i] for valid j indices  
        bb_ptrs = bb_ptr + j_idx * N + i
        bb_vals = tl.load(bb_ptrs, mask=valid_j_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        new_vals = prev_vals * prev_vals + bb_vals
        
        # Store result for valid j indices
        curr_ptrs = aa_ptr + j_idx * N + i
        tl.store(curr_ptrs, new_vals, mask=valid_j_mask)

def s232_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s232_kernel[grid](
        aa, bb, N, BLOCK_SIZE
    )
    
    return aa
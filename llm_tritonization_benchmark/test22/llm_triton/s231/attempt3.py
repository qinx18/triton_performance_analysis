import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Parallelize over i dimension
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Sequential loop over j dimension (dependency requires ordering)
    for j in range(1, N):
        # Current row addresses for aa[j][i] and bb[j][i]
        curr_aa_ptrs = aa_ptr + j * N + i_idx
        curr_bb_ptrs = bb_ptr + j * N + i_idx
        
        # Previous row addresses for aa[j-1][i]  
        prev_aa_ptrs = aa_ptr + (j - 1) * N + i_idx
        
        # Load data
        prev_aa_vals = tl.load(prev_aa_ptrs, mask=i_mask, other=0.0)
        bb_vals = tl.load(curr_bb_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = prev_aa_vals + bb_vals
        
        # Store result
        tl.store(curr_aa_ptrs, result, mask=i_mask)

def s231_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s231_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)
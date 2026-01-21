import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    for j in range(1, N):
        # Load aa[j-1][i]
        prev_offsets = (j - 1) * N + i_idx
        aa_prev = tl.load(aa_ptr + prev_offsets, mask=i_mask, other=0.0)
        
        # Load bb[j][i]  
        bb_offsets = j * N + i_idx
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = aa_prev + bb_vals
        
        # Store aa[j][i]
        aa_offsets = j * N + i_idx
        tl.store(aa_ptr + aa_offsets, result, mask=i_mask)

def s231_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Flatten tensors to ensure contiguous memory layout
    aa_flat = aa.contiguous().view(-1)
    bb_flat = bb.contiguous().view(-1)
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s231_kernel[grid](aa_flat, bb_flat, N, BLOCK_SIZE=BLOCK_SIZE)
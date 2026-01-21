import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(aa, bb, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < N
    
    for i in range(1, N):
        i_mask = i <= j_idx
        combined_mask = j_mask & i_mask
        
        # Load aa[j][i-1]
        prev_offsets = j_idx * N + (i - 1)
        aa_prev = tl.load(aa + prev_offsets, mask=combined_mask, other=0.0)
        
        # Load bb[j][i]
        bb_offsets = j_idx * N + i
        bb_vals = tl.load(bb + bb_offsets, mask=combined_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        result = aa_prev * aa_prev + bb_vals
        
        # Store result
        aa_offsets = j_idx * N + i
        tl.store(aa + aa_offsets, result, mask=combined_mask)

def s232_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process j from 1 to N-1
    j_size = N - 1
    grid = (triton.cdiv(j_size, BLOCK_SIZE),)
    
    s232_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)
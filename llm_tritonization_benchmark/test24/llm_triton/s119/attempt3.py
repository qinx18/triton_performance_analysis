import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, i_val: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate j indices for this block
    j_start = pid * BLOCK_SIZE + 1
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_vals = j_start + j_offsets
    
    # Mask for valid j values
    mask = j_vals < N
    
    # Load aa[i-1][j-1]
    aa_read_idx = (i_val - 1) * N + (j_vals - 1)
    aa_read_vals = tl.load(aa_ptr + aa_read_idx, mask=mask, other=0.0)
    
    # Load bb[i][j]
    bb_idx = i_val * N + j_vals
    bb_vals = tl.load(bb_ptr + bb_idx, mask=mask, other=0.0)
    
    # Compute aa[i][j] = aa[i-1][j-1] + bb[i][j]
    result = aa_read_vals + bb_vals
    
    # Store result
    aa_write_idx = i_val * N + j_vals
    tl.store(aa_ptr + aa_write_idx, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process row by row (i from 1 to N-1)
    for i in range(1, N):
        # Number of j elements to process (from 1 to N-1)
        num_j = N - 1
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE
        )
    
    return aa
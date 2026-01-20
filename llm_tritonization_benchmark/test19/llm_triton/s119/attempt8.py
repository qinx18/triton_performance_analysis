import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, i: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_start = pid * BLOCK_SIZE + 1
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    # Load aa[i-1][j-1] values
    prev_indices = (i - 1) * N + (j_offsets - 1)
    aa_prev = tl.load(aa_ptr + prev_indices, mask=j_mask)
    
    # Load bb[i][j] values
    bb_indices = i * N + j_offsets
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
    
    # Compute result
    result = aa_prev + bb_vals
    
    # Store to aa[i][j]
    out_indices = i * N + j_offsets
    tl.store(aa_ptr + out_indices, result, mask=j_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process row by row sequentially (due to i-dimension dependency)
    for i in range(1, N):
        # Parallelize j dimension for this row
        grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
        s119_kernel[grid](aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa
import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, LEN_2D, i_val, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets for j dimension
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    j_mask = j_offsets < LEN_2D
    
    # Vectorized load from aa[i_val-1][j-1] 
    aa_read_offsets = (i_val - 1) * LEN_2D + (j_offsets - 1)
    aa_read_mask = j_mask
    aa_vals = tl.load(aa_ptr + aa_read_offsets, mask=aa_read_mask, other=0.0)
    
    # Vectorized load from bb[i_val][j]
    bb_read_offsets = i_val * LEN_2D + j_offsets
    bb_vals = tl.load(bb_ptr + bb_read_offsets, mask=j_mask, other=0.0)
    
    # Vectorized computation
    result = aa_vals + bb_vals
    
    # Vectorized store to aa[i_val][j]
    aa_write_offsets = i_val * LEN_2D + j_offsets
    tl.store(aa_ptr + aa_write_offsets, result, mask=j_mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Sequential loop over i dimension
    for i_val in range(1, LEN_2D):
        # Number of j values to process (LEN_2D - 1)
        num_j = LEN_2D - 1
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, LEN_2D, i_val, BLOCK_SIZE=BLOCK_SIZE
        )
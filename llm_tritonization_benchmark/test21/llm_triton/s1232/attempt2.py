import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j in range(LEN_2D):
        i_idx = pid * BLOCK_SIZE + i_offsets + j
        
        # Only process valid indices within bounds
        mask = i_idx < LEN_2D
        
        # Calculate 2D indices for row-major layout
        aa_indices = i_idx * LEN_2D + j
        bb_indices = i_idx * LEN_2D + j
        cc_indices = i_idx * LEN_2D + j
        
        # Load bb and cc values
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask)
        cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + aa_indices, aa_vals, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(LEN_2D):
        # For triangular loop: only process i >= j
        valid_i = (i_idx >= j) & (i_idx < LEN_2D)
        
        # Calculate flat indices for aa[i][j], bb[i][j], cc[i][j]
        flat_idx = i_idx * LEN_2D + j
        
        # Load bb and cc values
        bb_vals = tl.load(bb_ptr + flat_idx, mask=valid_i, other=0.0)
        cc_vals = tl.load(cc_ptr + flat_idx, mask=valid_i, other=0.0)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + flat_idx, aa_vals, mask=valid_i)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Grid size based on i dimension
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
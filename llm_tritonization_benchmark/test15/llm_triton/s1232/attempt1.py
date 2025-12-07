import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(LEN_2D):
        # Mask for valid i indices: i >= j and i < LEN_2D
        mask = (i_idx >= j) & (i_idx < LEN_2D)
        
        # Calculate memory addresses for aa[i][j], bb[i][j], cc[i][j]
        aa_addr = aa_ptr + i_idx * LEN_2D + j
        bb_addr = bb_ptr + i_idx * LEN_2D + j
        cc_addr = cc_ptr + i_idx * LEN_2D + j
        
        # Load bb[i][j] and cc[i][j] values
        bb_vals = tl.load(bb_addr, mask=mask)
        cc_vals = tl.load(cc_addr, mask=mask)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        result = bb_vals + cc_vals
        
        # Store result to aa[i][j]
        tl.store(aa_addr, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
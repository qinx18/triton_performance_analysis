import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    
    for i in range(1, LEN_2D):
        j_mask = (j_idx < i) & (j_idx >= 0)
        
        # Load aa[j, i] (transpose access)
        aa_ji_ptrs = aa_ptr + j_idx * LEN_2D + i
        aa_ji = tl.load(aa_ji_ptrs, mask=j_mask, other=0.0)
        
        # Load bb[i, j]
        bb_ij_ptrs = bb_ptr + i * LEN_2D + j_idx
        bb_ij = tl.load(bb_ij_ptrs, mask=j_mask, other=0.0)
        
        # Compute aa[i, j] = aa[j, i] + bb[i, j]
        result = aa_ji + bb_ij
        
        # Store to aa[i, j]
        aa_ij_ptrs = aa_ptr + i * LEN_2D + j_idx
        tl.store(aa_ij_ptrs, result, mask=j_mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s114_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_base = pid * BLOCK_SIZE
    j_idx = j_base + j_offsets
    
    for i in range(LEN_2D):
        j_mask = (j_idx < i) & (j_idx >= 0)
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Read aa[j][i] - transpose read
            aa_ji_ptrs = aa_ptr + j_idx * LEN_2D + i
            aa_ji_vals = tl.load(aa_ji_ptrs, mask=j_mask, other=0.0)
            
            # Read bb[i][j]
            bb_ij_ptrs = aa_ptr + i * LEN_2D + j_idx
            bb_ij_vals = tl.load(bb_ptr + i * LEN_2D + j_idx, mask=j_mask, other=0.0)
            
            # Compute aa[i][j] = aa[j][i] + bb[i][j]
            result = aa_ji_vals + bb_ij_vals
            
            # Store aa[i][j]
            aa_ij_ptrs = aa_ptr + i * LEN_2D + j_idx
            tl.store(aa_ij_ptrs, result, mask=j_mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s114_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
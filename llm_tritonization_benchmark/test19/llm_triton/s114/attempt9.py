import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(LEN_2D):
        j_base = pid * BLOCK_SIZE
        j_idx = j_base + j_offsets
        
        # Only process j values where j < i
        valid_mask = (j_idx < i) & (j_idx < LEN_2D)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            # Calculate memory offsets for aa[i][j], aa[j][i], bb[i][j]
            aa_ij_offsets = i * LEN_2D + j_idx
            aa_ji_offsets = j_idx * LEN_2D + i
            bb_ij_offsets = i * LEN_2D + j_idx
            
            # Load aa[j][i] and bb[i][j]
            aa_ji_vals = tl.load(aa_ptr + aa_ji_offsets, mask=valid_mask, other=0.0)
            bb_ij_vals = tl.load(bb_ptr + bb_ij_offsets, mask=valid_mask, other=0.0)
            
            # Compute aa[i][j] = aa[j][i] + bb[i][j]
            result = aa_ji_vals + bb_ij_vals
            
            # Store to aa[i][j]
            tl.store(aa_ptr + aa_ij_offsets, result, mask=valid_mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, LEN_2D, BLOCK_SIZE)
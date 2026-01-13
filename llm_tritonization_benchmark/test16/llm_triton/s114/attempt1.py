import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(LEN_2D):
        j_start = pid * BLOCK_SIZE
        j_indices = j_start + j_offsets
        
        # Mask for valid j indices and triangular condition j < i
        mask = (j_indices < i) & (j_indices < LEN_2D)
        
        if tl.sum(mask.to(tl.int32)) > 0:
            # Load aa[j][i] (transposed access)
            aa_ji_ptrs = aa_ptr + j_indices * LEN_2D + i
            aa_ji = tl.load(aa_ji_ptrs, mask=mask)
            
            # Load bb[i][j]
            bb_ij_ptrs = aa_ptr + i * LEN_2D + j_indices
            bb_ij = tl.load(bb_ptr + i * LEN_2D + j_indices, mask=mask)
            
            # Compute aa[i][j] = aa[j][i] + bb[i][j]
            result = aa_ji + bb_ij
            
            # Store aa[i][j]
            aa_ij_ptrs = aa_ptr + i * LEN_2D + j_indices
            tl.store(aa_ij_ptrs, result, mask=mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s114_kernel[grid](
        aa, bb, LEN_2D, BLOCK_SIZE
    )
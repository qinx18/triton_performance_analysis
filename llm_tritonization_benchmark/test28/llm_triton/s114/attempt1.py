import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    
    for i in range(len_2d):
        j_mask = (j_idx < i) & (j_idx < len_2d)
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Load aa[j, i] (transpose indices)
            aa_ji_ptrs = aa_ptr + j_idx * len_2d + i
            aa_ji = tl.load(aa_ji_ptrs, mask=j_mask, other=0.0)
            
            # Load bb[i, j]
            bb_ij_ptrs = aa_ptr + i * len_2d + j_idx
            bb_ij = tl.load(bb_ptr + i * len_2d + j_idx, mask=j_mask, other=0.0)
            
            # Compute aa[i, j] = aa[j, i] + bb[i, j]
            result = aa_ji + bb_ij
            
            # Store aa[i, j]
            aa_ij_ptrs = aa_ptr + i * len_2d + j_idx
            tl.store(aa_ij_ptrs, result, mask=j_mask)

def s114_triton(aa, bb, len_2d):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s114_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
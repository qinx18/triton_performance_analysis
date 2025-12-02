import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential implementation to avoid race conditions
    for i in range(LEN_2D):
        # Process row i in blocks
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, i, BLOCK_SIZE):
            current_j = j_start + j_offsets
            j_mask = (current_j < i) & (current_j >= 0)
            
            # Load aa[j][i] (transpose access)
            aa_ji_ptrs = aa_ptr + current_j * LEN_2D + i
            aa_ji_vals = tl.load(aa_ji_ptrs, mask=j_mask, other=0.0)
            
            # Load bb[i][j]
            bb_ij_ptrs = bb_ptr + i * LEN_2D + current_j
            bb_ij_vals = tl.load(bb_ij_ptrs, mask=j_mask, other=0.0)
            
            # Compute aa[i][j] = aa[j][i] + bb[i][j]
            result = aa_ji_vals + bb_ij_vals
            
            # Store to aa[i][j]
            aa_ij_ptrs = aa_ptr + i * LEN_2D + current_j
            tl.store(aa_ij_ptrs, result, mask=j_mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    grid = (1,)
    s114_kernel[grid](aa, bb, LEN_2D, BLOCK_SIZE)
    
    return aa
import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program handles one row i
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Process elements in blocks for this row
    for j_start in range(0, i, BLOCK_SIZE):
        # Create offset vector once
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        
        # Mask for valid j indices (j < i)
        j_mask = j_indices < i
        
        # Calculate memory offsets for aa[i][j] and aa[j][i]
        aa_ij_offsets = i * LEN_2D + j_indices
        aa_ji_offsets = j_indices * LEN_2D + i
        bb_ij_offsets = i * LEN_2D + j_indices
        
        # Load aa[j][i] and bb[i][j]
        aa_ji_vals = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask, other=0.0)
        bb_ij_vals = tl.load(bb_ptr + bb_ij_offsets, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[j][i] + bb[i][j]
        result = aa_ji_vals + bb_ij_vals
        
        # Store result to aa[i][j]
        tl.store(aa_ptr + aa_ij_offsets, result, mask=j_mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one program per row
    BLOCK_SIZE = 128
    grid = (LEN_2D,)
    
    s114_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa
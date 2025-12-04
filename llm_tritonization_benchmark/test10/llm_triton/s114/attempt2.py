import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles one row
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Define offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements j = 0 to j < i in blocks
    for j_start in range(0, i, BLOCK_SIZE):
        j_offsets = j_start + offsets
        j_mask = (j_offsets < i) & (j_offsets >= 0)
        
        # Load aa[j][i] (transpose read)
        aa_ji_offsets = j_offsets * LEN_2D + i
        aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask, other=0.0)
        
        # Load bb[i][j]
        bb_ij_offsets = i * LEN_2D + j_offsets
        bb_ij = tl.load(bb_ptr + bb_ij_offsets, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[j][i] + bb[i][j]
        result = aa_ji + bb_ij
        
        # Store aa[i][j]
        aa_ij_offsets = i * LEN_2D + j_offsets
        tl.store(aa_ptr + aa_ij_offsets, result, mask=j_mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread per row
    grid = (LEN_2D,)
    
    s114_kernel[grid](
        aa, bb, 
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa
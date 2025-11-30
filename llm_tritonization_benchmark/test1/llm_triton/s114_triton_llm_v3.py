import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential processing of triangular matrix
    # Each program handles one row i
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Process all j < i for this row i
    for j_start in range(0, i, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, i)
        j_range = j_end - j_start
        
        if j_range <= 0:
            continue
            
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < j_end
        
        # Load bb[i][j] values
        bb_i_offsets = i * LEN_2D + j_offsets
        bb_vals = tl.load(bb_ptr + bb_i_offsets, mask=mask, other=0.0)
        
        # Load aa[j][i] values (transposed access)
        aa_ji_offsets = j_offsets * LEN_2D + i
        aa_ji_vals = tl.load(aa_ptr + aa_ji_offsets, mask=mask, other=0.0)
        
        # Compute aa[i][j] = aa[j][i] + bb[i][j]
        result = aa_ji_vals + bb_vals
        
        # Store to aa[i][j]
        aa_ij_offsets = i * LEN_2D + j_offsets
        tl.store(aa_ptr + aa_ij_offsets, result, mask=mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one program per row
    grid = (LEN_2D,)
    
    s114_kernel[grid](
        aa, bb, 
        LEN_2D, BLOCK_SIZE
    )
    
    return aa
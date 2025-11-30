import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(
    aa_ptr,
    aa_copy_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for j dimension
    j = tl.program_id(0) + 1
    
    if j >= LEN_2D:
        return
    
    # Sequential processing of i dimension to preserve dependencies
    for i in range(1, LEN_2D):
        # Calculate offsets for aa[j][i-1] and aa[j-1][i]
        offset_j_i_minus_1 = j * LEN_2D + (i - 1)
        offset_j_minus_1_i = (j - 1) * LEN_2D + i
        offset_j_i = j * LEN_2D + i
        
        # Load values from read-only copy
        val_j_i_minus_1 = tl.load(aa_copy_ptr + offset_j_i_minus_1)
        val_j_minus_1_i = tl.load(aa_copy_ptr + offset_j_minus_1_i)
        
        # Compute result
        result = (val_j_i_minus_1 + val_j_minus_1_i) / 1.9
        
        # Store to original array
        tl.store(aa_ptr + offset_j_i, result)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Launch kernel with one thread per j index (excluding j=0)
    grid = (LEN_2D - 1,)
    BLOCK_SIZE = 1
    
    s2111_kernel[grid](
        aa,
        aa_copy,
        LEN_2D,
        BLOCK_SIZE,
    )
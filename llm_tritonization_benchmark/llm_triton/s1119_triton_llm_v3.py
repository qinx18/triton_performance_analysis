import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for j dimension (parallelized)
    j_block_id = tl.program_id(0)
    j_start = j_block_id * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Sequential loop over i dimension (1 to LEN_2D-1)
    for i in range(1, LEN_2D):
        # Calculate read offsets for aa[i-1][j] (from copy)
        read_offsets = (i - 1) * LEN_2D + j_offsets
        
        # Calculate write offsets for aa[i][j] (to original)
        write_offsets = i * LEN_2D + j_offsets
        
        # Calculate bb offsets for bb[i][j]
        bb_offsets = i * LEN_2D + j_offsets
        
        # Load values
        aa_prev_vals = tl.load(aa_copy_ptr + read_offsets, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Compute result
        result = aa_prev_vals + bb_vals
        
        # Store result
        tl.store(aa_ptr + write_offsets, result, mask=j_mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    # Create read-only copy for WAR race condition handling
    aa_copy = aa.clone()
    
    # Calculate grid size for j dimension
    grid_j = triton.cdiv(LEN_2D, BLOCK_SIZE)
    grid = (grid_j,)
    
    s1119_kernel[grid](
        aa,
        aa_copy,
        bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
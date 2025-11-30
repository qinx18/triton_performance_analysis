import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get column index
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i < LEN_2D
    
    # Check condition: aa[0][i] > 0
    aa_0_offset = i
    aa_0_vals = tl.load(aa_copy_ptr + aa_0_offset, mask=i_mask, other=0.0)
    condition = aa_0_vals > 0.0
    
    # Process each column that meets the condition
    for j in range(1, LEN_2D):
        # Calculate offsets for j and j-1 rows
        offset_j = j * LEN_2D + i
        offset_j_prev = (j - 1) * LEN_2D + i
        
        # Load values from read-only copy
        aa_prev = tl.load(aa_copy_ptr + offset_j_prev, mask=i_mask & condition, other=0.0)
        bb_curr = tl.load(bb_ptr + offset_j, mask=i_mask & condition, other=0.0)
        cc_curr = tl.load(cc_ptr + offset_j, mask=i_mask & condition, other=0.0)
        
        # Compute: aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_prev + bb_curr * cc_curr
        
        # Store to original array
        tl.store(aa_ptr + offset_j, result, mask=i_mask & condition)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Calculate grid size
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    # Launch kernel
    s275_kernel[grid](
        aa,
        aa_copy,
        bb,
        cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return aa
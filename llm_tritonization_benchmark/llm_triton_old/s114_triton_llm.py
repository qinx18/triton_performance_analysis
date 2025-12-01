import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(
    aa_ptr, bb_ptr, output_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for lower triangular matrix update.
    Each program handles one row of the lower triangular matrix.
    """
    # Get the row index this program handles
    row_idx = tl.program_id(0)
    
    if row_idx >= LEN_2D:
        return
    
    # Process elements in blocks along the column dimension
    for col_start in range(0, row_idx, BLOCK_SIZE):
        # Calculate column indices for this block
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        
        # Mask to ensure we only process valid columns (j < i)
        col_mask = col_offsets < row_idx
        
        # Calculate memory offsets
        aa_offsets = row_idx * LEN_2D + col_offsets
        aa_t_offsets = col_offsets * LEN_2D + row_idx  # Transposed indices
        bb_offsets = row_idx * LEN_2D + col_offsets
        
        # Load data with masking
        aa_transposed = tl.load(aa_ptr + aa_t_offsets, mask=col_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=col_mask, other=0.0)
        
        # Perform computation: aa[i][j] = aa[j][i] + bb[i][j]
        result = aa_transposed + bb_vals
        
        # Store result
        tl.store(output_ptr + aa_offsets, result, mask=col_mask)

def s114_triton(aa, bb):
    """
    Triton implementation of TSVC s114 - lower triangular matrix update.
    
    Optimizations:
    - Row-parallel processing to handle lower triangular pattern efficiently
    - Block-based column processing to improve memory coalescing
    - Separate output tensor to avoid read-after-write hazards
    - Masking to handle triangular boundaries and edge cases
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    LEN_2D = aa.shape[0]
    
    # Create output tensor, initialize with original aa values
    output = aa.clone()
    
    # Launch kernel configuration
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    # Launch kernel
    s114_kernel[grid](
        aa, bb, output,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
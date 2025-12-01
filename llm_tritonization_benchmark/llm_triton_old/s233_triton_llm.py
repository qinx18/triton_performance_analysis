import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D, i_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s233 - processes one column at a time with sequential dependencies
    """
    pid = tl.program_id(0)
    
    # Each program handles a block of rows for the current column i_col
    row_start = pid * BLOCK_SIZE + 1  # Start from row 1
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < LEN_2D
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    # Process sequentially within each block due to dependency
    for local_j in range(BLOCK_SIZE):
        j = row_start + local_j
        if j < LEN_2D:
            aa_idx = j * LEN_2D + i_col
            aa_prev_idx = (j - 1) * LEN_2D + i_col
            cc_idx = j * LEN_2D + i_col
            
            aa_prev_val = tl.load(aa_ptr + aa_prev_idx)
            cc_val = tl.load(cc_ptr + cc_idx)
            result = aa_prev_val + cc_val
            tl.store(aa_ptr + aa_idx, result)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    # Load data for all rows in this block
    bb_curr_ptrs = bb_ptr + row_offsets * LEN_2D + i_col
    bb_prev_ptrs = bb_ptr + row_offsets * LEN_2D + (i_col - 1)
    cc_ptrs = cc_ptr + row_offsets * LEN_2D + i_col
    
    bb_prev_vals = tl.load(bb_prev_ptrs, mask=mask)
    cc_vals = tl.load(cc_ptrs, mask=mask)
    results = bb_prev_vals + cc_vals
    
    tl.store(bb_curr_ptrs, results, mask=mask)

def s233_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s233 with column-wise processing
    """
    aa = aa.contiguous()
    bb = bb.contiguous() 
    cc = cc.contiguous()
    
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Process each column sequentially due to aa dependencies
    for i in range(1, LEN_2D):
        # Calculate grid size for current column
        num_rows = LEN_2D - 1  # Process rows 1 to LEN_2D-1
        grid_size = triton.cdiv(num_rows, BLOCK_SIZE)
        
        # Launch kernel for current column
        s233_kernel[(grid_size,)](
            aa, bb, cc,
            LEN_2D, i,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return aa, bb
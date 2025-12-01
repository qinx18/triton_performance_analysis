import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(
    aa_ptr, bb_ptr, cc_ptr, result_ptr,
    len_2d,
    BLOCK_SIZE: tl.constexpr
):
    # Get column index
    col_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = col_idx < len_2d
    
    # Load first row to check condition
    first_row_offset = col_idx
    first_row_vals = tl.load(aa_ptr + first_row_offset, mask=col_mask, other=0.0)
    
    # Check condition: aa[0][col] > 0
    condition = first_row_vals > 0.0
    
    # Process each column that meets the condition
    for col in range(BLOCK_SIZE):
        if col_idx[col] < len_2d and condition[col]:
            # Initialize first element for this column
            prev_val = first_row_vals[col]
            tl.store(result_ptr + col_idx[col], prev_val, mask=col_idx[col] < len_2d)
            
            # Forward recurrence for this column
            for j in range(1, len_2d):
                bb_offset = j * len_2d + col_idx[col]
                cc_offset = j * len_2d + col_idx[col]
                result_offset = j * len_2d + col_idx[col]
                
                if col_idx[col] < len_2d:
                    bb_val = tl.load(bb_ptr + bb_offset)
                    cc_val = tl.load(cc_ptr + cc_offset)
                    new_val = prev_val + bb_val * cc_val
                    tl.store(result_ptr + result_offset, new_val)
                    prev_val = new_val

def s275_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s275 - conditional forward recurrence.
    Optimized for GPU execution with column-parallel processing.
    """
    aa = aa.contiguous()
    bb = bb.contiguous() 
    cc = cc.contiguous()
    
    # Clone aa to avoid modifying input and initialize result
    result_aa = aa.clone()
    
    # Get dimensions
    len_2d = aa.shape[1]
    
    # Handle empty case
    if len_2d == 0:
        return result_aa
    
    # Use smaller block size for better load balancing with conditional execution
    BLOCK_SIZE = 32
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    # Launch kernel - processes columns in parallel
    # Each thread block handles multiple columns and performs sequential recurrence
    s275_kernel[grid](
        aa, bb, cc, result_aa,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result_aa
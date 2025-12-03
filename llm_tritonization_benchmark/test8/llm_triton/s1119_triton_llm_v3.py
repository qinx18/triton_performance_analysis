import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, i_val: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Calculate memory offsets for 2D arrays
    curr_row_offset = i_val * LEN_2D + offsets
    prev_row_offset = (i_val - 1) * LEN_2D + offsets
    
    # Load values from previous row of aa and current row of bb
    aa_prev = tl.load(aa_ptr + prev_row_offset, mask=mask)
    bb_curr = tl.load(bb_ptr + curr_row_offset, mask=mask)
    
    # Compute: aa[i][j] = aa[i-1][j] + bb[i][j]
    result = aa_prev + bb_curr
    
    # Store result to current row of aa
    tl.store(aa_ptr + curr_row_offset, result, mask=mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i dimension (i starts from 1)
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1119_kernel[grid](
            aa, bb,
            LEN_2D=LEN_2D,
            i_val=i_val,
            BLOCK_SIZE=BLOCK_SIZE
        )
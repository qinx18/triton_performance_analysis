import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get column index
    col_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = col_idx < LEN_2D
    
    # Check condition: aa[0][col_idx] > 0
    aa_0_offset = col_idx
    condition_vals = tl.load(aa_ptr + aa_0_offset, mask=col_mask, other=0.0)
    condition_mask = condition_vals > 0.0
    
    # For each valid column, perform the sequential computation
    for col in range(BLOCK_SIZE):
        if col_idx[col] < LEN_2D:
            col_offset = col_idx[col]
            # Check if condition is met for this column
            aa_0_val = tl.load(aa_ptr + col_offset)
            if aa_0_val > 0.0:
                # Sequential loop: aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
                for j in range(1, LEN_2D):
                    prev_offset = (j - 1) * LEN_2D + col_offset
                    curr_offset = j * LEN_2D + col_offset
                    
                    prev_val = tl.load(aa_ptr + prev_offset)
                    bb_val = tl.load(bb_ptr + curr_offset)
                    cc_val = tl.load(cc_ptr + curr_offset)
                    
                    new_val = prev_val + bb_val * cc_val
                    tl.store(aa_ptr + curr_offset, new_val)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(32, LEN_2D)
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID for column (i dimension)
    pid = tl.program_id(0)
    col_idx = pid
    
    if col_idx >= LEN_2D:
        return
    
    # Check condition: aa[0][col_idx] > 0
    aa_0_ptr = aa_ptr + col_idx
    condition_val = tl.load(aa_0_ptr)
    
    if condition_val > 0.0:
        # Sequential loop for j dimension (WAR dependency)
        for j in range(1, LEN_2D):
            # Load aa[j-1][col_idx]
            aa_prev_ptr = aa_ptr + (j-1) * LEN_2D + col_idx
            aa_prev_val = tl.load(aa_prev_ptr)
            
            # Load bb[j][col_idx] and cc[j][col_idx]
            bb_ptr_curr = bb_ptr + j * LEN_2D + col_idx
            cc_ptr_curr = cc_ptr + j * LEN_2D + col_idx
            bb_val = tl.load(bb_ptr_curr)
            cc_val = tl.load(cc_ptr_curr)
            
            # Compute and store aa[j][col_idx] = aa[j-1][col_idx] + bb[j][col_idx] * cc[j][col_idx]
            result = aa_prev_val + bb_val * cc_val
            aa_curr_ptr = aa_ptr + j * LEN_2D + col_idx
            tl.store(aa_curr_ptr, result)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread block per column
    grid = (LEN_2D,)
    BLOCK_SIZE = 1
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D, BLOCK_SIZE
    )
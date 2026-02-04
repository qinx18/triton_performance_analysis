import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process anti-diagonals sequentially inside kernel
    for diag in range(2, 2 * N):
        j_vals = diag - i_offsets
        i_vals = i_offsets
        
        mask = (j_vals >= 1) & (j_vals < N) & (i_vals >= 1) & (i_vals < N)
        
        # Load aa[j-1][i] (previous row, same column)
        prev_row_offset = (j_vals - 1) * N + i_vals
        aa_prev_row = tl.load(aa_ptr + prev_row_offset, mask=mask, other=0.0)
        
        # Load aa[j][i-1] (same row, previous column)
        prev_col_offset = j_vals * N + (i_vals - 1)
        aa_prev_col = tl.load(aa_ptr + prev_col_offset, mask=mask, other=0.0)
        
        # Compute new value
        result = (aa_prev_row + aa_prev_col) / 1.9
        
        # Store aa[j][i]
        current_offset = j_vals * N + i_vals
        tl.store(aa_ptr + current_offset, result, mask=mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = min(256, N)
    
    grid = (1,)
    s2111_kernel[grid](aa, N=N, BLOCK_SIZE=BLOCK_SIZE)
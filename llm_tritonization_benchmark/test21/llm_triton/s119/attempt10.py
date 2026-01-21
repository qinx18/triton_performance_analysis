import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, diag_offset: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # For diagonal processing: i starts from 1 + block_start, j = diag_offset - i
    i_vals = 1 + block_start + offsets
    j_vals = diag_offset - i_vals
    
    # Valid range: 1 <= i < N, 1 <= j < N
    mask = (i_vals >= 1) & (j_vals >= 1) & (i_vals < N) & (j_vals < N)
    
    # Load bb[i][j]
    bb_indices = i_vals * N + j_vals
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Load aa[i-1][j-1]
    aa_prev_indices = (i_vals - 1) * N + (j_vals - 1)
    aa_prev_vals = tl.load(aa_ptr + aa_prev_indices, mask=mask, other=0.0)
    
    # Compute and store aa[i][j] = aa[i-1][j-1] + bb[i][j]
    result_vals = aa_prev_vals + bb_vals
    aa_indices = i_vals * N + j_vals
    tl.store(aa_ptr + aa_indices, result_vals, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process anti-diagonals sequentially (i + j = constant)
    for diag_sum in range(3, 2*N - 1):  # i + j ranges from 3 to 2*N-2
        # For this diagonal, i ranges from max(1, diag_sum-N+1) to min(diag_sum-1, N-1)
        max_elements = min(diag_sum - 1, N - 1) - max(1, diag_sum - N + 1) + 1
        
        if max_elements > 0:
            grid = (triton.cdiv(max_elements, BLOCK_SIZE),)
            s119_kernel[grid](
                aa, bb, N, diag_sum, BLOCK_SIZE
            )
    
    return aa
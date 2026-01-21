import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, diag: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Calculate i based on diagonal index
    i = 1 + idx
    j = diag - i
    
    # Valid range: i >= 1, j >= 1, i < N, j < N
    mask = (i >= 1) & (j >= 1) & (i < N) & (j < N)
    
    # Load bb[i][j]
    bb_indices = i * N + j
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Load aa[i-1][j-1]
    aa_prev_indices = (i - 1) * N + (j - 1)
    aa_prev_vals = tl.load(aa_ptr + aa_prev_indices, mask=mask, other=0.0)
    
    # Compute and store aa[i][j] = aa[i-1][j-1] + bb[i][j]
    result_vals = aa_prev_vals + bb_vals
    aa_indices = i * N + j
    tl.store(aa_ptr + aa_indices, result_vals, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2*N - 1):  # diag = i + j
        # Calculate number of valid elements on this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N - 1)
        num_elements = end_i - start_i + 1
        
        if num_elements > 0:
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            s119_kernel[grid](
                aa, bb, N, diag, BLOCK_SIZE
            )
    
    return aa
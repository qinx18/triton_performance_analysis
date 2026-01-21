import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, i_val: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_vals = 1 + j_offset
    mask = j_vals < N
    
    # Load bb values
    bb_indices = i_val * N + j_vals
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Load aa values from previous diagonal
    aa_prev_indices = (i_val - 1) * N + (j_vals - 1)
    aa_prev_vals = tl.load(aa_ptr + aa_prev_indices, mask=mask, other=0.0)
    
    # Compute and store results
    result_vals = aa_prev_vals + bb_vals
    aa_indices = i_val * N + j_vals
    tl.store(aa_ptr + aa_indices, result_vals, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential processing of i dimension
    for i in range(1, N):
        num_j_elements = N - 1  # j goes from 1 to N-1
        grid = (triton.cdiv(num_j_elements, BLOCK_SIZE),)
        s119_kernel[grid](
            aa, bb, N, i, BLOCK_SIZE
        )
    
    return aa
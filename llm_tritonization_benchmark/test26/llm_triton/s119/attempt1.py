import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, diag, start_i, end_i, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < (end_i - start_i)) & (indices >= 0)
    i_vals = start_i + indices
    j_vals = diag - i_vals
    
    # Check bounds
    valid_mask = mask & (i_vals >= 1) & (i_vals < N) & (j_vals >= 1) & (j_vals < N)
    
    # Load aa[i-1][j-1]
    aa_prev_offsets = (i_vals - 1) * N + (j_vals - 1)
    aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=valid_mask, other=0.0)
    
    # Load bb[i][j]
    bb_offsets = i_vals * N + j_vals
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=valid_mask, other=0.0)
    
    # Compute aa[i][j] = aa[i-1][j-1] + bb[i][j]
    result = aa_prev + bb_vals
    
    # Store result
    aa_offsets = i_vals * N + j_vals
    tl.store(aa_ptr + aa_offsets, result, mask=valid_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Wavefront parallelization - process anti-diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        if start_i >= end_i:
            continue
            
        num_elements = end_i - start_i
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, N, diag, start_i, end_i, BLOCK_SIZE
        )
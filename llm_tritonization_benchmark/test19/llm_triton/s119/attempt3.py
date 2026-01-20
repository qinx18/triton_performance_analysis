import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag, start_i, end_i, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = (indices < (end_i - start_i)) & (indices >= 0)
    
    i_vals = start_i + indices
    j_vals = diag - i_vals
    
    # Check bounds
    i_mask = (i_vals >= 1) & (i_vals < N) & (i_vals <= (N - 1))
    j_mask = (j_vals >= 1) & (j_vals < N) & (j_vals <= (N - 1))
    valid_mask = mask & i_mask & j_mask
    
    # Load aa[i-1][j-1] and bb[i][j]
    aa_read_offsets = (i_vals - 1) * N + (j_vals - 1)
    bb_read_offsets = i_vals * N + j_vals
    aa_write_offsets = i_vals * N + j_vals
    
    aa_vals = tl.load(aa_ptr + aa_read_offsets, mask=valid_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_read_offsets, mask=valid_mask, other=0.0)
    
    result = aa_vals + bb_vals
    
    tl.store(aa_ptr + aa_write_offsets, result, mask=valid_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        start_i = max(1, diag - (N - 1))
        end_i = min(diag - 1, N - 1)
        
        if start_i <= end_i:
            num_elements = end_i - start_i + 1
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            
            s119_kernel[grid](
                aa, bb, diag, start_i, end_i + 1, N, BLOCK_SIZE=BLOCK_SIZE
            )
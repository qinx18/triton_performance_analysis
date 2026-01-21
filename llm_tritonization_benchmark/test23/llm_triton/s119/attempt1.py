import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, diag_value, start_i, end_i, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < (end_i - start_i)) & (indices >= 0)
    valid_indices = indices + start_i
    
    i_vals = tl.where(mask, valid_indices, 0)
    j_vals = tl.where(mask, diag_value - valid_indices, 0)
    
    # Check bounds
    valid_mask = mask & (i_vals < N) & (j_vals < N) & (i_vals >= 1) & (j_vals >= 1)
    
    # Load aa[i-1][j-1]
    aa_read_idx = (i_vals - 1) * N + (j_vals - 1)
    aa_vals = tl.load(aa_ptr + aa_read_idx, mask=valid_mask, other=0.0)
    
    # Load bb[i][j]
    bb_read_idx = i_vals * N + j_vals
    bb_vals = tl.load(bb_ptr + bb_read_idx, mask=valid_mask, other=0.0)
    
    # Compute result
    result = aa_vals + bb_vals
    
    # Store aa[i][j]
    aa_write_idx = i_vals * N + j_vals
    tl.store(aa_ptr + aa_write_idx, result, mask=valid_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        if start_i >= end_i:
            continue
            
        num_elements = end_i - start_i
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, N, diag, start_i, end_i,
            BLOCK_SIZE=BLOCK_SIZE
        )
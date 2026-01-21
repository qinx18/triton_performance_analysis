import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag_idx, start_i, num_elements, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < num_elements
    
    # Calculate i, j from linear index within diagonal
    i_vals = start_i + indices
    j_vals = diag_idx - i_vals
    
    # Mask for valid i, j ranges (bounds checking)
    valid_mask = mask & (i_vals >= 1) & (i_vals < N) & (j_vals >= 1) & (j_vals < N)
    
    # Load bb[i][j]
    bb_indices = i_vals * N + j_vals
    bb_vals = tl.load(bb_ptr + bb_indices, mask=valid_mask, other=0.0)
    
    # Load aa[i-1][j-1]
    aa_prev_indices = (i_vals - 1) * N + (j_vals - 1)
    aa_prev_vals = tl.load(aa_ptr + aa_prev_indices, mask=valid_mask, other=0.0)
    
    # Compute result
    result = aa_prev_vals + bb_vals
    
    # Store aa[i][j]
    aa_indices = i_vals * N + j_vals
    tl.store(aa_ptr + aa_indices, result, mask=valid_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j, starting from 2 (i=1,j=1)
        # Calculate bounds for this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N - 1)
        
        if start_i > end_i:
            continue
            
        num_elements = end_i - start_i + 1
        
        if num_elements > 0:
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            s119_kernel[grid](
                aa, bb, diag, start_i, num_elements, N, BLOCK_SIZE=BLOCK_SIZE
            )
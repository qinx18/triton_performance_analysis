import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag, start_i, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = (i_offsets < (min(diag - 1, N - 1) - start_i + 1)) & (i_offsets >= 0)
    i_vals = start_i + i_offsets
    j_vals = diag - i_vals
    
    valid_mask = mask & (i_vals >= 1) & (i_vals < N) & (j_vals >= 1) & (j_vals < N)
    
    # Calculate indices
    curr_indices = i_vals * N + j_vals
    prev_indices = (i_vals - 1) * N + (j_vals - 1)
    
    # Load values
    aa_prev = tl.load(aa_ptr + prev_indices, mask=valid_mask, other=0.0)
    bb_curr = tl.load(bb_ptr + curr_indices, mask=valid_mask, other=0.0)
    
    # Compute and store
    result = aa_prev + bb_curr
    tl.store(aa_ptr + curr_indices, result, mask=valid_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        start_i = max(1, diag - N + 1)
        end_i = min(diag - 1, N - 1)
        
        if start_i <= end_i:
            num_elements = end_i - start_i + 1
            num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
            
            if num_blocks > 0:
                grid = (num_blocks,)
                s119_kernel[grid](
                    aa, bb, diag, start_i, N, BLOCK_SIZE
                )
    
    return aa
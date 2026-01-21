import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, start_i: tl.constexpr, num_elements: tl.constexpr, diag: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    valid_offsets = tl.where(mask, offsets, 0)
    i_vals = start_i + valid_offsets
    j_vals = diag - i_vals
    
    # Load bb values
    bb_indices = i_vals * N + j_vals
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Load aa values from previous diagonal
    aa_prev_indices = (i_vals - 1) * N + (j_vals - 1)
    aa_prev_vals = tl.load(aa_ptr + aa_prev_indices, mask=mask, other=0.0)
    
    # Compute and store results
    result_vals = aa_prev_vals + bb_vals
    aa_indices = i_vals * N + j_vals
    tl.store(aa_ptr + aa_indices, result_vals, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64  # Smaller block size to reduce memory pressure
    
    # Process anti-diagonals from 2 to 2*N-1
    for diag in range(2, 2 * N):
        # Calculate the range of i values for this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N - 1)
        
        if start_i > end_i:
            continue
            
        num_elements = end_i - start_i + 1
        
        if num_elements > 0:
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            s119_kernel[grid](
                aa, bb, N, start_i, num_elements, diag, BLOCK_SIZE
            )
    
    return aa
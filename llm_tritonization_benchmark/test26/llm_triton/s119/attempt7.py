import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, diag_val, start_i, BLOCK_SIZE: tl.constexpr):
    # Get block ID
    block_id = tl.program_id(0)
    
    # Calculate block start within this diagonal
    block_start = block_id * BLOCK_SIZE
    
    # Calculate offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = start_i + block_start + offsets
    j_indices = diag_val - i_indices
    
    # Create mask for valid elements
    mask = (i_indices < N) & (j_indices >= 1) & (j_indices < N) & (i_indices >= 1)
    
    # Calculate linear indices for current position
    aa_curr_idx = i_indices * N + j_indices
    # Calculate linear indices for previous position (i-1, j-1)
    aa_prev_idx = (i_indices - 1) * N + (j_indices - 1)
    bb_idx = i_indices * N + j_indices
    
    # Load values
    aa_prev = tl.load(aa_ptr + aa_prev_idx, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_idx, mask=mask, other=0.0)
    
    # Compute and store
    result = aa_prev + bb_vals
    tl.store(aa_ptr + aa_curr_idx, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j, from 2 to 2*N-1
        # Calculate range of i values for this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag - 1, N - 1)  # j must be >= 1, so i <= diag - 1
        
        num_elements = end_i - start_i + 1
        
        if num_elements <= 0:
            continue
        
        # Number of blocks needed for this diagonal
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        
        # Launch kernel for this diagonal
        grid = (num_blocks,)
        s119_kernel[grid](
            aa, bb, N, diag, start_i, BLOCK_SIZE
        )
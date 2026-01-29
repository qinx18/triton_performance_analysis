import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, diag, N, BLOCK_SIZE: tl.constexpr):
    # Get position within diagonal
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    pos_offsets = block_start + offsets
    
    # Calculate j indices for this diagonal
    # For diagonal diag, j ranges from max(1, diag-N+1) to min(diag-1, N-1)
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag - 1, N - 1)
    
    # Calculate actual j values for this block
    j_vals = start_j + pos_offsets
    i_vals = diag - j_vals
    
    # Create mask for valid positions
    valid_bounds = (j_vals >= 1) & (j_vals < N) & (i_vals >= 1) & (i_vals < N)
    valid_diag = (j_vals >= start_j) & (j_vals <= end_j)
    mask = valid_bounds & valid_diag
    
    # Calculate linear indices for 2D array access
    current_idx = j_vals * N + i_vals
    left_idx = j_vals * N + (i_vals - 1)
    top_idx = (j_vals - 1) * N + i_vals
    
    # Load values
    left_vals = tl.load(aa_ptr + left_idx, mask=mask, other=0.0)
    top_vals = tl.load(aa_ptr + top_idx, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + top_vals) / 1.9
    
    # Store results
    tl.store(aa_ptr + current_idx, new_vals, mask=mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 128
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        # Calculate range of j values for this diagonal
        start_j = max(1, diag - N + 1)
        end_j = min(diag - 1, N - 1)
        num_elements = end_j - start_j + 1
        
        if num_elements <= 0:
            continue
            
        # Calculate grid size
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        
        # Launch kernel for this diagonal
        s2111_kernel[(num_blocks,)](
            aa,
            diag,
            N,
            BLOCK_SIZE
        )
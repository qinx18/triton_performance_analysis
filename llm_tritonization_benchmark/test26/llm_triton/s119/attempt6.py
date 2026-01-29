import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the diagonal
    diag_id = tl.program_id(0)
    
    # Calculate actual diagonal value (diag = i + j)
    diag = diag_id + 2  # diagonals start from 2 (i=1, j=1)
    
    # Calculate range of i values for this diagonal
    start_i = max(1, diag - N + 1)
    end_i = min(diag, N)
    
    # Number of elements in this diagonal
    num_elements = end_i - start_i
    
    if num_elements <= 0:
        return
    
    # Block processing within diagonal
    block_start = tl.program_id(1) * BLOCK_SIZE
    
    if block_start >= num_elements:
        return
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = start_i + block_start + offsets
    j_indices = diag - i_indices
    
    # Create mask for valid elements
    mask = (block_start + offsets < num_elements) & (i_indices < N) & (j_indices < N) & (j_indices >= 1)
    
    # Calculate linear indices
    aa_curr_idx = i_indices * N + j_indices
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
    
    # Process diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j, from 2 to 2*N-1
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        num_elements = end_i - start_i
        
        if num_elements <= 0:
            continue
        
        # Number of blocks needed for this diagonal
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        
        # Launch kernel for this diagonal
        grid = (1, num_blocks)
        s119_kernel[grid](
            aa, bb, N, BLOCK_SIZE
        )
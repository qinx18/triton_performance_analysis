import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one anti-diagonal at a time
    diag = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    
    # Calculate actual diagonal index (2 to 2*N-1)
    actual_diag = diag + 2
    
    # Calculate bounds for this diagonal
    start_i = tl.maximum(1, actual_diag - N + 1)
    end_i = tl.minimum(actual_diag, N)
    
    # Calculate offsets within this diagonal
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Convert diagonal-relative indices to (i, j) coordinates
    i_vals = start_i + indices
    j_vals = actual_diag - i_vals
    
    # Create mask for valid elements
    mask = (indices < (end_i - start_i)) & (i_vals < N) & (j_vals < N) & (i_vals >= 1) & (j_vals >= 1)
    
    # Calculate linear indices
    aa_indices = i_vals * N + j_vals
    bb_indices = i_vals * N + j_vals
    aa_dep_indices = (i_vals - 1) * N + (j_vals - 1)
    
    # Load values
    aa_dep_vals = tl.load(aa_ptr + aa_dep_indices, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Compute result
    result = aa_dep_vals + bb_vals
    
    # Store result
    tl.store(aa_ptr + aa_indices, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    for diag in range(2 * N - 3):  # diag indices from 0 to 2*N-4 (actual diagonals 2 to 2*N-1)
        actual_diag = diag + 2
        start_i = max(1, actual_diag - N + 1)
        end_i = min(actual_diag, N)
        num_elements = end_i - start_i
        
        if num_elements > 0:
            BLOCK_SIZE = 256
            grid = (1, triton.cdiv(num_elements, BLOCK_SIZE))
            
            s119_kernel[grid](
                aa, bb, N,
                BLOCK_SIZE=BLOCK_SIZE
            )
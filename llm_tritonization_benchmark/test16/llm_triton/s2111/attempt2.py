import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N, diag, start_j, BLOCK_SIZE: tl.constexpr):
    # Get the current block position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = start_j + block_start + offsets
    
    # Calculate corresponding i indices
    i_indices = diag - j_indices
    
    # Mask for valid elements in this block
    mask = (j_indices < N) & (i_indices < N) & (j_indices >= 1) & (i_indices >= 1)
    
    # Load values needed for computation
    # aa[j][i-1] values
    left_offsets = j_indices * N + (i_indices - 1)
    left_vals = tl.load(aa_ptr + left_offsets, mask=mask, other=0.0)
    
    # aa[j-1][i] values  
    up_offsets = (j_indices - 1) * N + i_indices
    up_vals = tl.load(aa_ptr + up_offsets, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results
    result_offsets = j_indices * N + i_indices
    tl.store(aa_ptr + result_offsets, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j ranges from 2 to 2*N-1
        # Calculate the range of j values for this diagonal
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N - 1)
        
        # Number of elements in this diagonal
        num_elements = end_j - start_j + 1
        
        if num_elements > 0:
            # Calculate grid size
            grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
            
            # Launch kernel for this diagonal
            s2111_kernel[(grid_size,)](
                aa,
                N,
                diag,
                start_j,
                BLOCK_SIZE=BLOCK_SIZE
            )
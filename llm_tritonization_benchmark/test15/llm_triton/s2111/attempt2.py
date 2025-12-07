import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N, diag, start_j, BLOCK_SIZE: tl.constexpr):
    # Get block ID
    block_id = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = block_id * BLOCK_SIZE
    
    # Create offsets for elements in this block
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    
    # Calculate j coordinates for elements on this diagonal
    j_coords = start_j + element_ids
    
    # Calculate corresponding i coordinates (since j + i = diag)
    i_coords = diag - j_coords
    
    # Create mask for valid elements
    valid_mask = (element_ids < (diag - start_j)) & (j_coords < N) & (i_coords < N) & (j_coords >= 1) & (i_coords >= 1)
    
    # Calculate memory indices
    current_indices = j_coords * N + i_coords
    left_indices = j_coords * N + (i_coords - 1)  # aa[j][i-1]
    top_indices = (j_coords - 1) * N + i_coords   # aa[j-1][i]
    
    # Load values
    left_vals = tl.load(aa_ptr + left_indices, mask=valid_mask, other=0.0)
    top_vals = tl.load(aa_ptr + top_indices, mask=valid_mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + top_vals) / 1.9
    
    # Store results
    tl.store(aa_ptr + current_indices, new_vals, mask=valid_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * N - 1):
        # Calculate valid range of j for this diagonal
        start_j = max(1, diag - N + 1)
        end_j = min(diag - 1, N - 1)
        
        # Number of elements on this diagonal
        num_elements = end_j - start_j + 1
        
        if num_elements <= 0:
            continue
        
        # Calculate number of blocks needed
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        
        # Launch kernel for this diagonal
        s2111_kernel[(num_blocks,)](
            aa,
            N,
            diag,
            start_j,
            BLOCK_SIZE
        )
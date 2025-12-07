import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the diagonal
    diag_id = tl.program_id(0)
    
    # Calculate diagonal value: diag = j + i
    diag = diag_id + 2  # Start from diagonal 2 (since j,i both start from 1)
    
    # Calculate range of j values for this diagonal
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag, N - 1)
    
    # Number of elements on this diagonal
    num_elements = end_j - start_j + 1
    
    if num_elements <= 0:
        return
    
    # Block processing within the diagonal
    block_start = tl.program_id(1) * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    
    # Mask for valid elements
    mask = element_ids < num_elements
    
    # Calculate j and i coordinates for each element
    j_coords = start_j + element_ids
    i_coords = diag - j_coords
    
    # Additional bounds checking
    valid_coords = mask & (j_coords >= 1) & (j_coords < N) & (i_coords >= 1) & (i_coords < N)
    
    # Load values from aa[j][i-1] and aa[j-1][i]
    left_indices = j_coords * N + (i_coords - 1)
    top_indices = (j_coords - 1) * N + i_coords
    
    left_vals = tl.load(aa_ptr + left_indices, mask=valid_coords, other=0.0)
    top_vals = tl.load(aa_ptr + top_indices, mask=valid_coords, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + top_vals) / 1.9
    
    # Store results
    result_indices = j_coords * N + i_coords
    tl.store(aa_ptr + result_indices, new_vals, mask=valid_coords)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Maximum diagonal index (j + i can range from 2 to 2*N-2)
    max_diag = 2 * N - 2
    num_diagonals = max_diag - 1  # diagonals from 2 to 2*N-2
    
    BLOCK_SIZE = 256
    
    # Process each diagonal sequentially
    for diag_offset in range(num_diagonals):
        diag = diag_offset + 2
        
        # Calculate number of elements on this diagonal
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N - 1)
        num_elements = end_j - start_j + 1
        
        if num_elements <= 0:
            continue
            
        # Calculate grid size for this diagonal
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        
        # Launch kernel for this diagonal
        grid = (1, num_blocks)
        s2111_kernel[grid](
            aa,
            N,
            BLOCK_SIZE
        )
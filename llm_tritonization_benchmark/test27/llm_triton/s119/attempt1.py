import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this diagonal
    pid = tl.program_id(0)
    
    # Calculate diagonal index (diag = i + j)
    diag = pid + 2
    
    # Calculate range of i values for this diagonal
    start_i = tl.maximum(1, diag - N + 1)
    end_i = tl.minimum(diag, N)
    
    # Number of elements in this diagonal
    num_elements = end_i - start_i
    
    if num_elements <= 0:
        return
    
    # Create offset vector for elements in this diagonal
    offsets = tl.arange(0, BLOCK_SIZE)
    i_vals = start_i + offsets
    j_vals = diag - i_vals
    
    # Mask for valid elements
    mask = (offsets < num_elements) & (i_vals < N) & (j_vals < N) & (i_vals >= 1) & (j_vals >= 1)
    
    # Calculate linear indices for aa and bb arrays
    aa_indices = i_vals * N + j_vals
    bb_indices = i_vals * N + j_vals
    aa_dep_indices = (i_vals - 1) * N + (j_vals - 1)
    
    # Load values
    aa_dep_vals = tl.load(aa_ptr + aa_dep_indices, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask)
    
    # Compute new values
    new_vals = aa_dep_vals + bb_vals
    
    # Store results
    tl.store(aa_ptr + aa_indices, new_vals, mask=mask)

def s119_triton(aa, bb, len_2d):
    N = len_2d
    
    # Calculate number of anti-diagonals
    num_diagonals = 2 * N - 3  # diagonals from 2 to 2*N-2
    
    BLOCK_SIZE = 128
    
    # Process each anti-diagonal sequentially
    for diag_offset in range(num_diagonals):
        diag = diag_offset + 2
        
        # Calculate number of elements in this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        num_elements = end_i - start_i
        
        if num_elements <= 0:
            continue
        
        # Launch kernel for this diagonal
        grid = (1,)
        s119_kernel[grid](
            aa, bb, N, BLOCK_SIZE
        )
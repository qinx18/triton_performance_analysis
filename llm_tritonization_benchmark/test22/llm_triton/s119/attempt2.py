import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag, start_i, N, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Generate offsets within the block
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = start_i + block_start + offsets
    
    # Calculate corresponding j indices
    j_indices = diag - i_indices
    
    # Create mask for valid indices
    mask = (i_indices < N) & (j_indices >= 1) & (j_indices < N) & (i_indices >= 1)
    
    # Load data with proper 2D indexing
    aa_read_offsets = (i_indices - 1) * N + (j_indices - 1)
    bb_offsets = i_indices * N + j_indices
    aa_write_offsets = i_indices * N + j_indices
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_read_offsets, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Compute result
    result = aa_vals + bb_vals
    
    # Store result
    tl.store(aa_ptr + aa_write_offsets, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * N):  # diag = i + j, from 2 to 2*N-1
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        num_elements = end_i - start_i
        if num_elements <= 0:
            continue
            
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, diag, start_i, N,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return aa
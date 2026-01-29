import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N, diag_value, BLOCK_SIZE: tl.constexpr):
    # Get block ID and thread offsets
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate starting position for this block
    block_start = block_id * BLOCK_SIZE
    
    # Calculate valid range for j on this diagonal
    start_j = tl.maximum(1, diag_value - N + 1)
    end_j = tl.minimum(diag_value - 1, N - 1)
    
    # Calculate j values for this block
    j_vals = start_j + block_start + offsets
    
    # Check if j values are valid for this diagonal
    mask = (j_vals <= end_j) & (j_vals >= start_j)
    
    # Calculate corresponding i values
    i_vals = diag_value - j_vals
    
    # Additional mask for valid i values
    mask = mask & (i_vals >= 1) & (i_vals < N)
    
    # Load aa[j][i-1]
    prev_i_offsets = j_vals * N + (i_vals - 1)
    aa_prev_i = tl.load(aa_ptr + prev_i_offsets, mask=mask, other=0.0)
    
    # Load aa[j-1][i]
    prev_j_offsets = (j_vals - 1) * N + i_vals
    aa_prev_j = tl.load(aa_ptr + prev_j_offsets, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = (aa_prev_i + aa_prev_j) / 1.9
    
    # Store results
    store_offsets = j_vals * N + i_vals
    tl.store(aa_ptr + store_offsets, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    BLOCK_SIZE = 128
    
    # Process diagonals from 2 to 2*N-2
    for diag_value in range(2, 2 * N - 1):
        # Calculate valid range for j on this diagonal
        start_j = max(1, diag_value - N + 1)
        end_j = min(diag_value - 1, N - 1)
        
        # Number of elements on this diagonal
        diag_length = max(0, end_j - start_j + 1)
        
        if diag_length > 0:
            # Number of blocks needed for this diagonal
            num_blocks = triton.cdiv(diag_length, BLOCK_SIZE)
            
            # Launch kernel for this diagonal
            grid = (num_blocks,)
            s2111_kernel[grid](
                aa,
                N,
                diag_value,
                BLOCK_SIZE
            )
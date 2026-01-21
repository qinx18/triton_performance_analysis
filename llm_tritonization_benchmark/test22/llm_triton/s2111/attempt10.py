import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, diag, start_j, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get block of j indices to process
    block_start = tl.program_id(0) * BLOCK_SIZE
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = start_j + block_start + j_offsets
    
    # Calculate corresponding i indices
    i_indices = diag - j_indices
    
    # Mask for valid indices
    j_mask = j_indices < N
    i_mask = (i_indices >= 1) & (i_indices < N)
    valid_mask = j_mask & i_mask
    
    # Load values from aa[j][i-1] and aa[j-1][i]
    left_offsets = j_indices * N + (i_indices - 1)
    up_offsets = (j_indices - 1) * N + i_indices
    
    val_left = tl.load(aa_ptr + left_offsets, mask=valid_mask, other=0.0)
    val_up = tl.load(aa_ptr + up_offsets, mask=valid_mask, other=0.0)
    
    # Compute result
    result = (val_left + val_up) / 1.9
    
    # Store result
    store_offsets = j_indices * N + i_indices
    tl.store(aa_ptr + store_offsets, result, mask=valid_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process each diagonal sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag + 1, N)
        
        if start_j < end_j:
            num_elements = end_j - start_j
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            s2111_kernel[grid](aa, diag, start_j, N=N, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa
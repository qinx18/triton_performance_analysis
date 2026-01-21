import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, diag: tl.constexpr, start_j: tl.constexpr, num_elements: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    elem_indices = block_start + offsets
    mask = elem_indices < num_elements
    
    # Convert element index to j coordinate
    j_indices = start_j + elem_indices
    i_indices = diag - j_indices
    
    # Check bounds
    valid_mask = mask & (j_indices < N) & (i_indices < N) & (j_indices >= 1) & (i_indices >= 1)
    
    # Load values: aa[j][i-1] and aa[j-1][i]
    left_offsets = j_indices * N + (i_indices - 1)
    up_offsets = (j_indices - 1) * N + i_indices
    
    left_vals = tl.load(aa_ptr + left_offsets, mask=valid_mask, other=0.0)
    up_vals = tl.load(aa_ptr + up_offsets, mask=valid_mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results: aa[j][i]
    out_offsets = j_indices * N + i_indices
    tl.store(aa_ptr + out_offsets, new_vals, mask=valid_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag - 1, N - 1)
        
        if start_j <= end_j:
            num_elements = end_j - start_j + 1
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            
            s2111_kernel[grid](
                aa, diag, start_j, num_elements, N, BLOCK_SIZE
            )
import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, diag_idx: tl.constexpr, start_j: tl.constexpr, num_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < num_elements
    
    # Convert linear indices to (j, i) coordinates on the diagonal
    j_coords = start_j + indices
    i_coords = diag_idx - j_coords
    
    # Valid coordinates check
    coord_mask = (j_coords >= 1) & (j_coords < N) & (i_coords >= 1) & (i_coords < N)
    final_mask = mask & coord_mask
    
    # Load aa[j][i-1] and aa[j-1][i]
    left_offsets = j_coords * N + (i_coords - 1)
    up_offsets = (j_coords - 1) * N + i_coords
    
    left_vals = tl.load(aa_ptr + left_offsets, mask=final_mask, other=0.0)
    up_vals = tl.load(aa_ptr + up_offsets, mask=final_mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results
    store_offsets = j_coords * N + i_coords
    tl.store(aa_ptr + store_offsets, new_vals, mask=final_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        start_j = max(1, diag - (N - 1))
        end_j = min(diag - 1, N - 1)
        num_elements = end_j - start_j + 1
        
        if num_elements <= 0:
            continue
        
        BLOCK_SIZE = 256
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, N, diag, start_j, num_elements, BLOCK_SIZE
        )
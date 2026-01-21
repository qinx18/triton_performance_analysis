import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, diag_sum: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # For diagonal sum k, j ranges from max(1, k-N+1) to min(k-1, N-1)
    j_min = tl.maximum(1, diag_sum - N + 1)
    j_max = tl.minimum(diag_sum - 1, N - 1)
    num_elements = j_max - j_min + 1
    
    mask = indices < num_elements
    
    # Compute j and i coordinates
    j_coords = j_min + indices
    i_coords = diag_sum - j_coords
    
    # Additional bounds check
    valid_mask = mask & (j_coords >= 1) & (j_coords < N) & (i_coords >= 1) & (i_coords < N)
    
    # Load aa[j][i-1] and aa[j-1][i]
    left_offsets = j_coords * N + (i_coords - 1)
    up_offsets = (j_coords - 1) * N + i_coords
    
    left_vals = tl.load(aa_ptr + left_offsets, mask=valid_mask, other=0.0)
    up_vals = tl.load(aa_ptr + up_offsets, mask=valid_mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results at aa[j][i]
    store_offsets = j_coords * N + i_coords
    tl.store(aa_ptr + store_offsets, new_vals, mask=valid_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    for diag_sum in range(2, 2 * N - 1):
        j_min = max(1, diag_sum - N + 1)
        j_max = min(diag_sum - 1, N - 1)
        num_elements = j_max - j_min + 1
        
        if num_elements <= 0:
            continue
        
        BLOCK_SIZE = 256
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, N, diag_sum, BLOCK_SIZE
        )
import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, diag, start_j, end_j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = start_j + block_start + offsets
    
    mask = (j_indices < end_j) & (j_indices >= start_j)
    
    valid_j = tl.where(mask, j_indices, 0)
    valid_i = diag - valid_j
    
    range_mask = mask & (valid_i >= 1) & (valid_i < N) & (valid_j >= 1) & (valid_j < N)
    
    # Load aa[j][i-1] and aa[j-1][i]
    left_idx = valid_j * N + (valid_i - 1)
    top_idx = (valid_j - 1) * N + valid_i
    
    left_vals = tl.load(aa_ptr + left_idx, mask=range_mask, other=0.0)
    top_vals = tl.load(aa_ptr + top_idx, mask=range_mask, other=0.0)
    
    # Compute new value
    new_vals = (left_vals + top_vals) / 1.9
    
    # Store result
    result_idx = valid_j * N + valid_i
    tl.store(aa_ptr + result_idx, new_vals, mask=range_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    # Process wavefront pattern: iterate over anti-diagonals
    for diag in range(2, 2 * N - 1):  # diag = i + j
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        if start_j >= end_j:
            continue
            
        num_elements = end_j - start_j
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa,
            diag,
            start_j,
            end_j,
            N,
            BLOCK_SIZE
        )
import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, diag, start_j, end_j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    if block_start >= (end_j - start_j):
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = start_j + block_start + offsets
    mask = (j_offsets < end_j) & (j_offsets >= start_j)
    
    # Compute corresponding i values: i = diag - j
    i_offsets = diag - j_offsets
    
    # Additional bounds check
    valid_mask = mask & (i_offsets >= 1) & (i_offsets < N) & (j_offsets >= 1) & (j_offsets < N)
    
    # Load aa[j-1][i] (row-major: (j-1) * N + i)
    prev_j_idx = (j_offsets - 1) * N + i_offsets
    aa_prev_j = tl.load(aa_ptr + prev_j_idx, mask=valid_mask, other=0.0)
    
    # Load aa[j][i-1] (row-major: j * N + (i-1))
    prev_i_idx = j_offsets * N + (i_offsets - 1)
    aa_prev_i = tl.load(aa_ptr + prev_i_idx, mask=valid_mask, other=0.0)
    
    # Compute new values: (aa[j][i-1] + aa[j-1][i]) / 1.9
    new_vals = (aa_prev_i + aa_prev_j) / 1.9
    
    # Store results at aa[j][i] (row-major: j * N + i)
    current_idx = j_offsets * N + i_offsets
    tl.store(aa_ptr + current_idx, new_vals, mask=valid_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2*N - 1):
        start_j = max(1, diag - (N - 1))
        end_j = min(diag - 1 + 1, N)
        
        if start_j >= end_j:
            continue
        
        num_elements = end_j - start_j
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](aa, diag, start_j, end_j, N, BLOCK_SIZE)
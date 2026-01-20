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
    i_indices = diag - j_indices
    valid_mask = mask & (i_indices >= 1) & (i_indices < N) & (j_indices >= 1) & (j_indices < N)
    
    # Load aa[j-1][i]
    prev_j_offsets = (j_indices - 1) * N + i_indices
    aa_prev_j = tl.load(aa_ptr + prev_j_offsets, mask=valid_mask, other=0.0)
    
    # Load aa[j][i-1]  
    prev_i_offsets = j_indices * N + (i_indices - 1)
    aa_prev_i = tl.load(aa_ptr + prev_i_offsets, mask=valid_mask, other=0.0)
    
    # Compute new value
    new_val = (aa_prev_j + aa_prev_i) / 1.9
    
    # Store aa[j][i]
    curr_offsets = j_indices * N + i_indices
    tl.store(aa_ptr + curr_offsets, new_val, mask=valid_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N - 1) + 1
        
        if start_j >= end_j:
            continue
            
        num_elements = end_j - start_j
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](aa, diag, start_j, end_j, N, BLOCK_SIZE=BLOCK_SIZE)
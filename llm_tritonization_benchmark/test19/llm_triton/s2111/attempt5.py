import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N, diag, start_j, end_j, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate j values for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_vals = start_j + pid * BLOCK_SIZE + j_offsets
    j_mask = (j_vals >= start_j) & (j_vals <= end_j)
    
    # Calculate corresponding i values: i = diag - j
    i_vals = diag - j_vals
    
    # Bounds check
    valid_mask = j_mask & (j_vals >= 1) & (j_vals < N) & (i_vals >= 1) & (i_vals < N)
    
    # Load values from aa[j][i-1] and aa[j-1][i]
    idx_left = j_vals * N + (i_vals - 1)  # aa[j][i-1]
    idx_up = (j_vals - 1) * N + i_vals    # aa[j-1][i]
    
    left_vals = tl.load(aa_ptr + idx_left, mask=valid_mask, other=0.0)
    up_vals = tl.load(aa_ptr + idx_up, mask=valid_mask, other=0.0)
    
    # Compute new values: (left + up) / 1.9
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results to aa[j][i]
    idx_current = j_vals * N + i_vals
    tl.store(aa_ptr + idx_current, new_vals, mask=valid_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2*N - 1):  # diag = i + j
        start_j = max(1, diag - (N - 1))
        end_j = min(diag - 1, N - 1)
        
        if start_j <= end_j:
            num_elements = end_j - start_j + 1
            BLOCK_SIZE = 64
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            
            s2111_kernel[grid](
                aa, N, diag, start_j, end_j, 
                BLOCK_SIZE=BLOCK_SIZE
            )
    
    return aa
import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, diag, start_j, end_j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate j for this block
    j = start_j + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values
    j_mask = (j >= start_j) & (j < end_j) & (j >= 1) & (j < N)
    
    # Calculate corresponding i values
    i = diag - j
    
    # Mask for valid i values
    i_mask = (i >= 1) & (i < N)
    
    # Combined mask
    mask = j_mask & i_mask
    
    # Load aa[j][i-1]
    idx_i_minus_1 = j * N + (i - 1)
    aa_i_minus_1 = tl.load(aa_ptr + idx_i_minus_1, mask=mask, other=0.0)
    
    # Load aa[j-1][i]
    idx_j_minus_1 = (j - 1) * N + i
    aa_j_minus_1 = tl.load(aa_ptr + idx_j_minus_1, mask=mask, other=0.0)
    
    # Compute new value
    new_val = (aa_i_minus_1 + aa_j_minus_1) / 1.9
    
    # Store result
    idx_current = j * N + i
    tl.store(aa_ptr + idx_current, new_val, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    BLOCK_SIZE = 128
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        if start_j >= end_j:
            continue
            
        num_elements = end_j - start_j
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, diag, start_j, end_j, N, BLOCK_SIZE=BLOCK_SIZE
        )
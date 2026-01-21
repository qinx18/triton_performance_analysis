import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, diag: tl.constexpr, 
                 BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Calculate j from diagonal index
    start_j = tl.maximum(1, diag - N + 1)
    j_vals = start_j + idx
    i_vals = diag - j_vals
    
    # Mask for valid indices
    j_mask = j_vals < tl.minimum(diag, N)
    i_mask = (i_vals >= 1) & (i_vals < N)
    mask = j_mask & i_mask
    
    # Load aa[j][i-1] values
    prev_i_indices = j_vals * N + (i_vals - 1)
    prev_i_vals = tl.load(aa_ptr + prev_i_indices, mask=mask, other=0.0)
    
    # Load aa[j-1][i] values
    prev_j_indices = (j_vals - 1) * N + i_vals
    prev_j_vals = tl.load(aa_ptr + prev_j_indices, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = (prev_i_vals + prev_j_vals) / 1.9
    
    # Store results
    store_indices = j_vals * N + i_vals
    tl.store(aa_ptr + store_indices, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    if not aa.is_contiguous():
        aa = aa.contiguous()
    
    BLOCK_SIZE = 32
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        if start_j >= end_j:
            continue
        
        num_elements = end_j - start_j
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, N=N, diag=diag, BLOCK_SIZE=BLOCK_SIZE
        )
    
    return aa
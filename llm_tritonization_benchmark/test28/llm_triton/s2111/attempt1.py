import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, diag, start_j, num_elements, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    element_offsets = block_start + offsets
    
    mask = element_offsets < num_elements
    
    j_vals = start_j + element_offsets
    i_vals = diag - j_vals
    
    valid_mask = mask & (j_vals < N) & (i_vals < N) & (j_vals >= 1) & (i_vals >= 1)
    
    # Load aa[j][i-1] and aa[j-1][i]
    left_idx = j_vals * N + (i_vals - 1)
    top_idx = (j_vals - 1) * N + i_vals
    
    left_vals = tl.load(aa_ptr + left_idx, mask=valid_mask, other=0.0)
    top_vals = tl.load(aa_ptr + top_idx, mask=valid_mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + top_vals) / 1.9
    
    # Store back to aa[j][i]
    current_idx = j_vals * N + i_vals
    tl.store(aa_ptr + current_idx, new_vals, mask=valid_mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        num_elements = end_j - start_j
        if num_elements <= 0:
            continue
            
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa,
            diag,
            start_j,
            num_elements,
            N,
            BLOCK_SIZE
        )
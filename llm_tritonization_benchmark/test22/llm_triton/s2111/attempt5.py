import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, diag: tl.constexpr, start_j: tl.constexpr, 
                 num_elements: tl.constexpr, N: tl.constexpr, 
                 BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    elem_idx = block_start + offsets
    
    # Mask for valid elements
    mask = elem_idx < num_elements
    
    # Convert element index to (j, i) coordinates
    j_vals = start_j + elem_idx
    i_vals = diag - j_vals
    
    # Additional mask for valid coordinates
    coord_mask = mask & (j_vals >= 1) & (j_vals < N) & (i_vals >= 1) & (i_vals < N)
    
    # Load aa[j][i-1] values
    prev_i_indices = j_vals * N + (i_vals - 1)
    prev_i_vals = tl.load(aa_ptr + prev_i_indices, mask=coord_mask, other=0.0)
    
    # Load aa[j-1][i] values  
    prev_j_indices = (j_vals - 1) * N + i_vals
    prev_j_vals = tl.load(aa_ptr + prev_j_indices, mask=coord_mask, other=0.0)
    
    # Compute new values
    new_vals = (prev_i_vals + prev_j_vals) / 1.9
    
    # Store results
    store_indices = j_vals * N + i_vals
    tl.store(aa_ptr + store_indices, new_vals, mask=coord_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    if not aa.is_contiguous():
        aa = aa.contiguous()
    
    BLOCK_SIZE = 256
    
    # Process anti-diagonals: elements where j + i = diag are independent
    for diag in range(2, 2*N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N - 1)
        num_elements = end_j - start_j + 1
        
        if num_elements > 0:
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            
            s2111_kernel[grid](
                aa, diag=diag, start_j=start_j, 
                num_elements=num_elements, N=N, BLOCK_SIZE=BLOCK_SIZE
            )
    
    return aa
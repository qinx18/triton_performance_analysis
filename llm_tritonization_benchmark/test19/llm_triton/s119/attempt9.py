import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag_val: tl.constexpr, start_i: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which i value this thread block handles
    i_offset = pid * BLOCK_SIZE
    i_val = start_i + i_offset
    
    # Create mask for valid i values
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = (start_i + i_offsets) < min(diag_val, N)
    
    # Calculate corresponding j values: j = diag - i
    i_vals = start_i + i_offsets
    j_vals = diag_val - i_vals
    
    # Additional mask for valid j values (should be >= 1 and < N)
    j_mask = (j_vals >= 1) & (j_vals < N)
    full_mask = i_mask & j_mask
    
    # Calculate array indices
    aa_read_indices = (i_vals - 1) * N + (j_vals - 1)  # aa[i-1][j-1]
    bb_read_indices = i_vals * N + j_vals              # bb[i][j]
    aa_write_indices = i_vals * N + j_vals             # aa[i][j]
    
    # Load values
    aa_prev = tl.load(aa_ptr + aa_read_indices, mask=full_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_read_indices, mask=full_mask, other=0.0)
    
    # Compute result
    result = aa_prev + bb_vals
    
    # Store result
    tl.store(aa_ptr + aa_write_indices, result, mask=full_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j, starting from 2 since i,j both start from 1
        # Calculate valid range of i for this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        if start_i >= end_i:
            continue
            
        num_elements = end_i - start_i
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s119_kernel[grid](aa, bb, diag, start_i, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa
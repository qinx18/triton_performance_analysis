import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag_val: tl.constexpr, start_i: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which elements this thread block handles
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    
    # Calculate i values for this block
    i_vals = start_i + element_ids
    
    # Calculate corresponding j values: j = diag - i
    j_vals = diag_val - i_vals
    
    # Create masks for valid indices
    valid_elements = element_ids < (diag_val - start_i)
    valid_i = (i_vals >= 1) & (i_vals < N)
    valid_j = (j_vals >= 1) & (j_vals < N)
    mask = valid_elements & valid_i & valid_j
    
    # Calculate array indices for 2D access
    aa_read_indices = (i_vals - 1) * N + (j_vals - 1)  # aa[i-1][j-1]
    bb_read_indices = i_vals * N + j_vals              # bb[i][j]
    aa_write_indices = i_vals * N + j_vals             # aa[i][j]
    
    # Load values
    aa_prev = tl.load(aa_ptr + aa_read_indices, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_read_indices, mask=mask, other=0.0)
    
    # Compute result
    result = aa_prev + bb_vals
    
    # Store result
    tl.store(aa_ptr + aa_write_indices, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process anti-diagonals sequentially (wavefront pattern)
    for diag in range(2, 2 * N):  # diag = i + j, starting from 2
        # Calculate valid range of i for this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        # Skip if no valid elements
        if start_i >= end_i:
            continue
            
        num_elements = end_i - start_i
        
        # Skip empty diagonals
        if num_elements <= 0:
            continue
            
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s119_kernel[grid](aa, bb, diag, start_i, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa
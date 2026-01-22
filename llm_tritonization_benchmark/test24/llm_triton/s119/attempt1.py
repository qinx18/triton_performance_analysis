import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, diag: tl.constexpr, start_i: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate the range of i values for this diagonal
    num_elements = diag - start_i
    if num_elements <= start_i:
        num_elements = N - start_i
    
    # Calculate which elements this program handles
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid elements
    mask = i_offsets < num_elements
    
    # Calculate actual i and j coordinates
    i_vals = start_i + i_offsets
    j_vals = diag - i_vals
    
    # Additional bounds checking
    valid_mask = mask & (i_vals >= 1) & (i_vals < N) & (j_vals >= 1) & (j_vals < N)
    
    # Load aa[i-1][j-1]
    aa_read_i = i_vals - 1
    aa_read_j = j_vals - 1
    aa_read_idx = aa_read_i * N + aa_read_j
    aa_read_vals = tl.load(aa_ptr + aa_read_idx, mask=valid_mask, other=0.0)
    
    # Load bb[i][j]
    bb_idx = i_vals * N + j_vals
    bb_vals = tl.load(bb_ptr + bb_idx, mask=valid_mask, other=0.0)
    
    # Compute aa[i][j] = aa[i-1][j-1] + bb[i][j]
    result = aa_read_vals + bb_vals
    
    # Store result
    aa_write_idx = i_vals * N + j_vals
    tl.store(aa_ptr + aa_write_idx, result, mask=valid_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        # Calculate the range of i values for this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        # Skip if no valid elements
        if start_i >= end_i:
            continue
            
        num_elements = end_i - start_i
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, N, diag, start_i, BLOCK_SIZE=BLOCK_SIZE
        )
    
    return aa
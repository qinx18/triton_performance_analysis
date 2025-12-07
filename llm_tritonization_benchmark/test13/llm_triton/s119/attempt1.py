import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Sequential computation due to wavefront dependencies
    # Each program handles one element sequentially
    pid = tl.program_id(0)
    
    # Calculate i, j from linear program id
    total_elements = (N - 1) * (N - 1)
    if pid >= total_elements:
        return
    
    # Convert linear index to 2D coordinates (1-based indexing)
    linear_idx = pid
    i = (linear_idx // (N - 1)) + 1
    j = (linear_idx % (N - 1)) + 1
    
    # Load aa[i-1][j-1] and bb[i][j]
    aa_prev_offset = (i - 1) * N + (j - 1)
    bb_curr_offset = i * N + j
    aa_curr_offset = i * N + j
    
    aa_prev_val = tl.load(aa_ptr + aa_prev_offset)
    bb_curr_val = tl.load(bb_ptr + bb_curr_offset)
    
    # Compute and store result
    result = aa_prev_val + bb_curr_val
    tl.store(aa_ptr + aa_curr_offset, result)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Due to wavefront dependencies, we need to process sequentially
    # Process by anti-diagonals to maintain correctness
    for diag in range(2, 2 * N):
        # Calculate range of i values for this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        if start_i >= end_i:
            continue
            
        # Process this diagonal in parallel
        elements_in_diag = end_i - start_i
        if elements_in_diag <= 0:
            continue
            
        BLOCK_SIZE = 64
        grid_size = triton.cdiv(elements_in_diag, 1)
        
        s119_diagonal_kernel[grid_size,](
            aa, bb, N, diag, start_i, elements_in_diag, BLOCK_SIZE=BLOCK_SIZE
        )

@triton.jit
def s119_diagonal_kernel(aa_ptr, bb_ptr, N, diag, start_i, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid >= num_elements:
        return
        
    # Calculate actual i, j coordinates
    i = start_i + pid
    j = diag - i
    
    # Bounds check
    if i >= N or j >= N or i < 1 or j < 1:
        return
    
    # Load aa[i-1][j-1] and bb[i][j]
    aa_prev_offset = (i - 1) * N + (j - 1)
    bb_curr_offset = i * N + j
    aa_curr_offset = i * N + j
    
    aa_prev_val = tl.load(aa_ptr + aa_prev_offset)
    bb_curr_val = tl.load(bb_ptr + bb_curr_offset)
    
    # Compute and store result
    result = aa_prev_val + bb_curr_val
    tl.store(aa_ptr + aa_curr_offset, result)
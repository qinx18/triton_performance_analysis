import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Process anti-diagonals sequentially, parallelize within each diagonal
    diag_idx = tl.program_id(0)
    thread_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate diagonal value: diag = j + i, ranges from 2 to 2*N-2
    diag = diag_idx + 2
    
    # Calculate range of j values for this diagonal
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag, N)
    num_elements = end_j - start_j
    
    # Check if this thread should process an element
    mask = thread_idx < num_elements
    
    # Calculate j and i coordinates
    j = start_j + thread_idx
    i = diag - j
    
    # Additional bounds check
    valid = mask & (j >= 1) & (j < N) & (i >= 1) & (i < N)
    
    # Load values
    left_offset = j * N + (i - 1)
    top_offset = (j - 1) * N + i
    current_offset = j * N + i
    
    left_val = tl.load(aa_ptr + left_offset, mask=valid)
    top_val = tl.load(aa_ptr + top_offset, mask=valid)
    
    # Compute new value
    new_val = (left_val + top_val) / 1.9
    
    # Store result
    tl.store(aa_ptr + current_offset, new_val, mask=valid)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Number of anti-diagonals to process (excluding first diagonal which starts at j=0,i=0)
    num_diagonals = 2 * N - 3  # diagonals 2 through 2*N-2
    
    # Process each anti-diagonal sequentially
    for diag_idx in range(num_diagonals):
        diag = diag_idx + 2
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        num_elements = end_j - start_j
        
        if num_elements > 0:
            grid = (triton.cdiv(num_elements, BLOCK_SIZE), 1)
            s2111_kernel[grid](aa, N, BLOCK_SIZE)
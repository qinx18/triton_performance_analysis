import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, diag: tl.constexpr, start_j: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate j coordinate for this block
    j = start_j + pid
    
    # Check if j is valid for this diagonal
    if j >= N or j < 1:
        return
    
    # Calculate corresponding i coordinate
    i = diag - j
    
    # Check bounds
    if i >= N or i < 1:
        return
    
    # Load values needed for computation
    # aa[j][i-1]
    left_offset = j * N + (i - 1)
    left_val = tl.load(aa_ptr + left_offset)
    
    # aa[j-1][i]
    top_offset = (j - 1) * N + i
    top_val = tl.load(aa_ptr + top_offset)
    
    # Compute new value
    new_val = (left_val + top_val) / 1.9
    
    # Store result
    current_offset = j * N + i
    tl.store(aa_ptr + current_offset, new_val)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        # Calculate valid j range for this diagonal
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N - 1)
        
        if start_j > end_j:
            continue
        
        # Number of elements on this diagonal
        num_elements = end_j - start_j + 1
        
        # Launch kernel with one thread per element on the diagonal
        BLOCK_SIZE = 256
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa,
            N=N,
            diag=diag,
            start_j=start_j,
            BLOCK_SIZE=BLOCK_SIZE
        )
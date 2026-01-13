import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, diag, start_j, end_j, N, BLOCK_SIZE: tl.constexpr):
    # Get program id for this block
    pid = tl.program_id(0)
    
    # Calculate which element on the diagonal this thread handles
    element_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j = start_j + element_idx
    
    # Calculate corresponding i for this diagonal
    i = diag - j
    
    # Check bounds
    valid = (j < end_j) & (i >= 1) & (i < N) & (j >= 1) & (j < N)
    
    # Load the two required values with masking
    left_val = tl.load(aa_ptr + j * N + (i - 1), mask=valid, other=0.0)  # aa[j][i-1]
    up_val = tl.load(aa_ptr + (j - 1) * N + i, mask=valid, other=0.0)    # aa[j-1][i]
    
    # Compute new value
    new_val = (left_val + up_val) / 1.9
    
    # Store result with masking
    tl.store(aa_ptr + j * N + i, new_val, mask=valid)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * N - 1):  # diag = i + j
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        # Number of elements on this diagonal
        num_elements = end_j - start_j
        
        if num_elements <= 0:
            continue
            
        # Launch kernel with one thread per element on the diagonal
        BLOCK_SIZE = 32
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa,
            diag,
            start_j, 
            end_j,
            N,
            BLOCK_SIZE
        )
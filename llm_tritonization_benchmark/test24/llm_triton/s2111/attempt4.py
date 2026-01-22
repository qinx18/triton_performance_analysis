import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, diag: tl.constexpr, start_j: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the position along the diagonal
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate j index along diagonal
    j = start_j + idx
    
    # Calculate i index: i = diag - j
    i = diag - j
    
    # Mask for valid indices
    mask = (j < N) & (i >= 1) & (i < N) & (j >= 1)
    
    # Calculate linear indices
    current_idx = j * N + i
    left_idx = j * N + (i - 1)
    up_idx = (j - 1) * N + i
    
    # Load values
    left_val = tl.load(aa_ptr + left_idx, mask=mask, other=0.0)
    up_val = tl.load(aa_ptr + up_idx, mask=mask, other=0.0)
    
    # Compute new value
    new_val = (left_val + up_val) / 1.9
    
    # Store result
    tl.store(aa_ptr + current_idx, new_val, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        # Calculate range of j values for this diagonal
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N - 1)
        
        if start_j <= end_j:
            num_elements = end_j - start_j + 1
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            
            s2111_kernel[grid](
                aa,
                diag=diag,
                start_j=start_j,
                N=N,
                BLOCK_SIZE=BLOCK_SIZE
            )
import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, diag: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate base offset for this thread block
    base_idx = pid * BLOCK_SIZE
    offsets = base_idx + tl.arange(0, BLOCK_SIZE)
    
    # For diagonal d, valid j range is max(1, d-N+1) to min(d-1, N-1)
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag - 1, N - 1)
    
    # Convert linear index to j coordinate
    j = start_j + offsets
    
    # Check if j is valid
    mask = (j <= end_j) & (j >= 1) & (j < N)
    
    # Calculate corresponding i coordinate
    i = diag - j
    
    # Additional bounds check for i
    mask = mask & (i >= 1) & (i < N)
    
    # Load aa[j][i-1]
    left_offsets = j * N + (i - 1)
    left_vals = tl.load(aa_ptr + left_offsets, mask=mask, other=0.0)
    
    # Load aa[j-1][i]
    top_offsets = (j - 1) * N + i
    top_vals = tl.load(aa_ptr + top_offsets, mask=mask, other=0.0)
    
    # Compute new value
    new_vals = (left_vals + top_vals) / 1.9
    
    # Store result
    current_offsets = j * N + i
    tl.store(aa_ptr + current_offsets, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        # Calculate valid j range for this diagonal
        start_j = max(1, diag - N + 1)
        end_j = min(diag - 1, N - 1)
        
        if start_j > end_j:
            continue
        
        # Number of elements on this diagonal
        num_elements = end_j - start_j + 1
        
        # Launch kernel
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa,
            N=N,
            diag=diag,
            BLOCK_SIZE=BLOCK_SIZE
        )
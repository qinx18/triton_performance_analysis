import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, diag: tl.constexpr, start_j: tl.constexpr, 
                 num_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < num_elements
    
    # Calculate j and i from linear index
    j = start_j + indices
    i = diag - j
    
    # Bounds checking
    valid_j = (j >= 1) & (j < N)
    valid_i = (i >= 1) & (i < N)
    valid = mask & valid_j & valid_i
    
    # Load values
    left_offsets = j * N + (i - 1)  # aa[j][i-1]
    up_offsets = (j - 1) * N + i    # aa[j-1][i]
    
    left_vals = tl.load(aa_ptr + left_offsets, mask=valid, other=0.0)
    up_vals = tl.load(aa_ptr + up_offsets, mask=valid, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results
    out_offsets = j * N + i
    tl.store(aa_ptr + out_offsets, new_vals, mask=valid)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        num_elements = end_j - start_j
        if num_elements <= 0:
            continue
            
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, N, diag, start_j, num_elements, BLOCK_SIZE
        )
import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute i indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector for i values (starting from j+1)
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + (j + 1)
    
    # Create mask for valid i values (i < N)
    mask = i_indices < N
    
    # Load aa[j][i] values
    aa_offsets = j * N + i_indices
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load current a[i] values
    a_vals = tl.load(a_ptr + i_indices, mask=mask)
    
    # Compute a[i] -= aa[j][i] * a[j]
    new_vals = a_vals - aa_vals * a_j
    
    # Store results back
    tl.store(a_ptr + i_indices, new_vals, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(N):
        # Number of i values to process (from j+1 to N-1)
        num_i = N - (j + 1)
        if num_i <= 0:
            continue
            
        # Launch kernel for this j
        grid = (triton.cdiv(num_i, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)
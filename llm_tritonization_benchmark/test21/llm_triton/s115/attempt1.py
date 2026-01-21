import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    # Calculate thread index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + (j + 1)
    
    # Create mask for valid i values (i >= j+1 and i < N)
    mask = i_offsets < N
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * N + i_offsets
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store results back
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel over i
    for j in range(N):
        # Number of i values to process (from j+1 to N-1)
        num_i = N - (j + 1)
        
        if num_i <= 0:
            continue
            
        # Launch kernel for this j
        grid = (triton.cdiv(num_i, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)
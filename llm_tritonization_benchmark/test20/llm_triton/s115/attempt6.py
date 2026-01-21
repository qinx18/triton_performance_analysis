import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    # Get block start and create offset vector
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + j + 1
    
    # Create mask for valid indices
    mask = indices < N
    
    # Load a[indices] values
    a_indices = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load aa[j][indices] values (row-major: aa[j][i] = aa[j*N + i])
    aa_indices = j * N + indices
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Load a[j] (scalar broadcasted)
    a_j = tl.load(a_ptr + j)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_indices - aa_vals * a_j
    
    # Store back to a[indices]
    tl.store(a_ptr + indices, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel over i for each j
    for j in range(N):
        # Number of elements to process: from j+1 to N-1
        num_elements = N - (j + 1)
        if num_elements <= 0:
            continue
            
        # Launch kernel with appropriate grid size
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)
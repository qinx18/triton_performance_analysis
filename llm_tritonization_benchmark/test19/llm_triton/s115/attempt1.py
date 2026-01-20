import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE + (j + 1)  # Start from j+1
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Mask for valid indices (i < N and i > j)
    mask = (i_indices < N) & (i_indices > j)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask)
    
    # Load aa[j][i] values (2D indexing: j * N + i)
    aa_ji_indices = j * N + i_indices
    aa_ji = tl.load(aa_ptr + aa_ji_indices, mask=mask)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]  # LEN_2D dimension
    
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel computation for each i
    for j in range(N):
        # Number of elements to process: from j+1 to N-1
        num_elements = N - (j + 1)
        if num_elements <= 0:
            continue
            
        # Calculate grid size
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        # Launch kernel
        s115_kernel[grid](
            a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE
        )
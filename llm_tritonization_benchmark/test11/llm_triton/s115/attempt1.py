import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate i starting position (j+1) for this block
    block_start = pid * BLOCK_SIZE + (j + 1)
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Mask for valid indices
    mask = i_indices < n
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_indices = j * n + i_indices
    aa_ji = tl.load(aa_ptr + aa_indices, mask=mask)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    n = aa.shape[0]  # LEN_2D
    BLOCK_SIZE = 256
    
    for j in range(n):
        # Calculate number of i elements to process (j+1 to n-1)
        num_elements = n - (j + 1)
        if num_elements <= 0:
            continue
            
        # Calculate grid size
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s115_kernel[grid](
            a, aa, j, n, BLOCK_SIZE
        )
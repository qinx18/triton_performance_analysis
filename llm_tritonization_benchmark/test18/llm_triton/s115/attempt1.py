import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Calculate the range of i values for this j: i in [j+1, LEN_2D)
    start_i = j + 1
    num_i = LEN_2D - start_i
    
    if num_i <= 0:
        return
    
    # Get block of i indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = start_i + block_start + offsets
    
    # Mask for valid i indices
    mask = (block_start + offsets) < num_i
    
    # Load a[j] (scalar, broadcast to vector)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * LEN_2D + i_indices
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values
    a_i_vals = tl.load(a_ptr + i_indices, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    new_a_i = a_i_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, new_a_i, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel execution over i
    for j in range(LEN_2D):
        # Calculate number of i values for this j
        start_i = j + 1
        num_i = LEN_2D - start_i
        
        if num_i <= 0:
            continue
            
        # Launch kernel with appropriate grid size
        grid_size = triton.cdiv(num_i, BLOCK_SIZE)
        
        s115_kernel[(grid_size,)](
            a, aa, j, LEN_2D, BLOCK_SIZE
        )
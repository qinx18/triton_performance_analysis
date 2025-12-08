import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Calculate the range of i values: j+1 to LEN_2D-1
    start_i = j + 1
    n_elements = LEN_2D - start_i
    
    if n_elements <= 0:
        return
    
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = start_i + block_start + offsets
    
    # Mask for valid i indices
    mask = (block_start + offsets < n_elements) & (i_offsets < LEN_2D)
    
    # Load a[j] (scalar, broadcast to vector)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * LEN_2D + i_offsets
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Sequential loop over j, parallel over i
    for j in range(LEN_2D):
        start_i = j + 1
        n_elements = LEN_2D - start_i
        
        if n_elements <= 0:
            continue
            
        # Launch kernel for this j
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        
        if num_blocks > 0:
            s115_kernel[(num_blocks,)](
                a, aa, j, LEN_2D, BLOCK_SIZE
            )
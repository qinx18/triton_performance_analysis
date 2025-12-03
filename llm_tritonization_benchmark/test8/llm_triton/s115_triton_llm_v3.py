import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute i indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Mask for valid i values (i must be < LEN_2D and i > j_val)
    mask = (i_indices < LEN_2D) & (i_indices > j_val)
    
    # Load a[j_val] once (scalar broadcast)
    a_j = tl.load(a_ptr + j_val)
    
    # Load a[i] values for all i in this block
    a_i_vals = tl.load(a_ptr + i_indices, mask=mask)
    
    # Load aa[j_val][i] values - aa is stored in row-major order
    aa_offsets = j_val * LEN_2D + i_indices
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j values
    for j in range(LEN_2D):
        # Calculate grid size for i dimension
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        
        # Launch kernel for this j value
        s115_kernel[grid](
            a, aa, j, 
            LEN_2D=LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )
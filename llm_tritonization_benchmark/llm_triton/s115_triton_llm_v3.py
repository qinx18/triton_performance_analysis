import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate i indices starting from j+1
    base_i = (j_val + 1) + pid * BLOCK_SIZE
    i_offsets = base_i + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid i values (i < LEN_2D)
    i_mask = i_offsets < LEN_2D
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j_val)
    
    # Load aa[j][i] values
    aa_offsets = j_val * LEN_2D + i_offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=i_mask)
    
    # Load a[i] values
    a_i_vals = tl.load(a_ptr + i_offsets, mask=i_mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i_vals - aa_vals * a_j
    
    # Store results back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j values
    for j in range(LEN_2D):
        # Number of i values for this j (i goes from j+1 to LEN_2D-1)
        num_i = LEN_2D - (j + 1)
        if num_i <= 0:
            continue
            
        grid = (triton.cdiv(num_i, BLOCK_SIZE),)
        s115_kernel[grid](
            a, aa, j, LEN_2D, BLOCK_SIZE
        )
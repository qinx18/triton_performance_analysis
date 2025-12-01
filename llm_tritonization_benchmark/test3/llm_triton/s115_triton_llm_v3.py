import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate i indices starting from j+1
    start_i = j_val + 1
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_indices = start_i + offsets
    
    # Mask for valid i values (i < LEN_2D)
    mask = i_indices < LEN_2D
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    
    # Load a[j] (scalar, broadcasted)
    a_j = tl.load(a_ptr + j_val)
    
    # Load aa[j][i] values
    aa_indices = j_val * LEN_2D + i_indices
    aa_ji = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(LEN_2D):
        # Number of i values to process (from j+1 to LEN_2D-1)
        num_i = LEN_2D - (j + 1)
        if num_i <= 0:
            continue
            
        grid = (triton.cdiv(num_i, BLOCK_SIZE),)
        s115_kernel[grid](
            a, aa.flatten(), j, LEN_2D, BLOCK_SIZE=BLOCK_SIZE
        )
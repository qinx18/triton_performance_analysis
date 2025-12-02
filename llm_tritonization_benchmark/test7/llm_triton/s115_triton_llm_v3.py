import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, len_2d, j_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Create offsets for i indices (j_val+1 to len_2d-1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_indices = j_val + 1 + offsets
    
    # Mask for valid i indices
    mask = i_indices < len_2d
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask)
    
    # Load aa[j][i] values (aa is row-major: aa[j][i] = aa[j*len_2d + i])
    aa_ji_indices = j_val * len_2d + i_indices
    aa_ji = tl.load(aa_ptr + aa_ji_indices, mask=mask)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j_val)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    len_2d = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(len_2d):
        # Number of i values for this j (j+1 to len_2d-1)
        num_i = len_2d - (j + 1)
        
        if num_i > 0:
            grid = (triton.cdiv(num_i, BLOCK_SIZE),)
            s115_kernel[grid](
                a, aa, len_2d, j, BLOCK_SIZE
            )
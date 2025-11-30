import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(
    a_ptr,
    aa_ptr,
    j_val,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate i indices starting from j+1
    i_start = j_val + 1
    i_base = i_start + pid * BLOCK_SIZE
    i_offsets = i_base + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid i indices
    i_mask = i_offsets < LEN_2D
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j_val)
    
    # Load aa[j][i] values
    aa_offsets = j_val * LEN_2D + i_offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=i_mask, other=0.0)

    # Load a[i] values
    a_i_vals = tl.load(a_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(LEN_2D):
        # Number of i elements to process (from j+1 to LEN_2D-1)
        num_i = LEN_2D - (j + 1)
        if num_i <= 0:
            continue
            
        # Launch kernel for this j value
        grid = (triton.cdiv(num_i, BLOCK_SIZE),)
        s115_kernel[grid](
            a,
            aa,
            j,
            LEN_2D,
            BLOCK_SIZE,
        )
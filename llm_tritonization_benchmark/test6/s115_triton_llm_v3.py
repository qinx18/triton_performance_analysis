import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(
    a_ptr,
    aa_ptr,
    j_val,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Create offset vector once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate i indices (i starts from j+1)
    i_offsets = pid * BLOCK_SIZE + offsets + (j_val + 1)
    
    # Mask for valid i indices
    i_mask = i_offsets < LEN_2D
    
    # Load a[j] (scalar value)
    a_j = tl.load(a_ptr + j_val)
    
    # Load aa[j][i] values
    aa_offsets = j_val * LEN_2D + i_offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=i_mask, other=0.0)
    
    # Load a[i] values
    a_i_vals = tl.load(a_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i_vals - aa_vals * a_j
    
    # Store result back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j values
    for j in range(LEN_2D):
        # Number of i values for this j (i ranges from j+1 to LEN_2D-1)
        num_i_vals = LEN_2D - (j + 1)
        if num_i_vals <= 0:
            continue
            
        # Calculate grid size for i parallelization
        grid = (triton.cdiv(num_i_vals, BLOCK_SIZE),)
        
        s115_kernel[grid](
            a,
            aa,
            j,
            LEN_2D,
            BLOCK_SIZE,
        )
import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate i range: j+1 to LEN_2D-1
    i_start = j + 1
    i_end = LEN_2D
    
    if i_start >= i_end:
        return
    
    # Block offset for i values
    block_start = pid * BLOCK_SIZE
    block_offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = i_start + block_start + block_offsets
    
    # Mask for valid i values
    mask = (i_offsets < i_end) & (i_offsets < LEN_2D)
    
    # Load a[j] once (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * LEN_2D + i_offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values
    a_i_vals = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(LEN_2D):
        # Calculate number of i values to process
        i_count = LEN_2D - (j + 1)
        if i_count <= 0:
            continue
            
        # Launch kernel for current j
        grid = (triton.cdiv(i_count, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, LEN_2D, BLOCK_SIZE)
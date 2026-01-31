import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + (j + 1)
    
    mask = (i_offsets < len_2d) & (i_offsets > j)
    
    # Load a[j]
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i_offsets]
    aa_offsets = j * len_2d + i_offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i_offsets]
    a_vals = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Compute a[i] -= aa[j][i] * a[j]
    new_vals = a_vals - aa_vals * a_j
    
    # Store back to a[i_offsets]
    tl.store(a_ptr + i_offsets, new_vals, mask=mask)

def s115_triton(a, aa, len_2d):
    BLOCK_SIZE = 256
    
    for j in range(len_2d):
        num_elements = len_2d - (j + 1)
        if num_elements <= 0:
            continue
            
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s115_kernel[grid](
            a, aa, j, len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )
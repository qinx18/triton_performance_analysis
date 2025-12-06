import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, LEN_2D, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = (j + 1) + block_id * BLOCK_SIZE + offsets
    
    mask = (i_offsets < LEN_2D) & (i_offsets > j)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * LEN_2D + i_offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(LEN_2D):
        num_elements = LEN_2D - (j + 1)
        if num_elements <= 0:
            continue
            
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        
        s115_kernel[(num_blocks,)](
            a, aa, j, LEN_2D, BLOCK_SIZE
        )
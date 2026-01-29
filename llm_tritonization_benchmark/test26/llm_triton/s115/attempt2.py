import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, n, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = (j + 1) + i_offsets
    
    mask = i_indices < n
    
    # Load a[j]
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * n + i_indices
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + i_indices, mask=mask)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    n = aa.shape[0]
    
    for j in range(n):
        num_i = n - (j + 1)
        if num_i <= 0:
            continue
            
        BLOCK_SIZE = 256
        num_blocks = triton.cdiv(num_i, BLOCK_SIZE)
        
        s115_kernel[(num_blocks,)](
            a, aa, j, n, BLOCK_SIZE
        )
import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(
    a_ptr,
    a_copy_ptr,
    aa_ptr,
    j,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE + (j + 1)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Load a[i] values from copy (read-only)
    a_i = tl.load(a_copy_ptr + offsets, mask=mask, other=0.0)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_copy_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * LEN_2D + offsets
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store result to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    # Sequential j loop, parallel i loop
    for j in range(LEN_2D):
        num_elements = LEN_2D - (j + 1)
        if num_elements <= 0:
            continue
            
        grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
        
        s115_kernel[(grid_size,)](
            a,
            a_copy,
            aa,
            j,
            LEN_2D,
            BLOCK_SIZE,
        )
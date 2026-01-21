import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, j: tl.constexpr, N: tl.constexpr, 
                 BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_vals = block_start + offsets + 1
    
    # Mask for valid indices
    mask = i_vals < N
    
    # Load aa[j][i-1] values
    prev_i_indices = j * N + (i_vals - 1)
    prev_i_vals = tl.load(aa_ptr + prev_i_indices, mask=mask, other=0.0)
    
    # Load aa[j-1][i] values  
    prev_j_indices = (j - 1) * N + i_vals
    prev_j_vals = tl.load(aa_ptr + prev_j_indices, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = (prev_i_vals + prev_j_vals) / 1.9
    
    # Store results
    store_indices = j * N + i_vals
    tl.store(aa_ptr + store_indices, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    if not aa.is_contiguous():
        aa = aa.contiguous()
    
    BLOCK_SIZE = 32
    
    # Process rows sequentially, parallelize within each row
    for j in range(1, N):
        num_elements = N - 1
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, j=j, N=N, BLOCK_SIZE=BLOCK_SIZE
        )
    
    return aa
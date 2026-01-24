import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    # Get block of indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1  # Start from i=1
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Create mask for valid indices (1 <= i < N)
    mask = i_offsets < N
    
    # Calculate memory offsets for aa[j][i] and aa[k][i-1]
    aa_j_offsets = j * N + i_offsets
    aa_k_offsets = k * N + (i_offsets - 1)
    
    # Load aa[k][i-1] values
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=mask)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load c[1] (scalar)
    c_1 = tl.load(c_ptr + 1)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store result
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    # Calculate number of elements to process (1 to N-1)
    n_elements = N - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
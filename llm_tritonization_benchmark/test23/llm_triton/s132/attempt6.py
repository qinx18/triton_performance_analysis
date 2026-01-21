import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust for loop starting at i=1
    i_vals = offsets + 1
    mask = i_vals < N
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_vals, mask=mask)
    
    # Load c[1] (scalar broadcast)
    c_1 = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] values - need i-1 indices
    i_minus_1 = i_vals - 1
    aa_k_indices = k * N + i_minus_1
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store to aa[j][i]
    aa_j_indices = j * N + i_vals
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    # We need to process indices 1 to N-1 (loop: i=1; i < N; i++)
    n_elements = N - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](aa, b, c, j, k, N, BLOCK_SIZE=BLOCK_SIZE)
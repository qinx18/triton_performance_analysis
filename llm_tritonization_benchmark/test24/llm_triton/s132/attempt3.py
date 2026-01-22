import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # i ranges from 1 to N-1, so adjust offsets to start from 1
    i_vals = offsets + 1
    mask = i_vals < N
    
    # Load c[1]
    c1 = tl.load(c_ptr + 1)
    
    # Load b[i] where i starts from 1
    b_vals = tl.load(b_ptr + i_vals, mask=mask)
    
    # Load aa[k][i-1] - when i starts from 1, i-1 starts from 0
    aa_k_indices = k * N + (i_vals - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1
    
    # Store to aa[j][i]
    aa_j_indices = j * N + i_vals
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    # Process i from 1 to N-1 (N-1 elements total)
    n_elements = N - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
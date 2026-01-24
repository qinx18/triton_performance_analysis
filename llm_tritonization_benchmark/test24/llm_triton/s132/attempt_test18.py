import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE + 1
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < N
    
    # Load c[1]
    c_val = tl.load(c_ptr + 1)
    
    # Load b[i]
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load aa[k][i-1]
    aa_k_offsets = k * N + (i_offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_val
    
    # Store to aa[j][i]
    aa_j_offsets = j * N + i_offsets
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    # Grid configuration - start from i=1, so process (N-1) elements
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(N - 1, BLOCK_SIZE)
    
    s132_kernel[(num_blocks,)](
        aa, b, c, j, k, N, BLOCK_SIZE
    )
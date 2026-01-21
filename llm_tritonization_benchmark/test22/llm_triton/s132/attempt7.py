import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # We need i from 1 to N-1, so adjust offsets
    i_offsets = offsets + 1
    mask = i_offsets < N
    
    # Load c[1] - single scalar value
    c_1 = tl.load(c_ptr + 1)
    
    # Load b[i] for current block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load aa[k][i-1] - this is aa[k][offsets] since i = offsets + 1
    aa_k_offsets = k * N + offsets
    mask_k = offsets < (N - 1)  # Valid range for i-1 when i goes from 1 to N-1
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=mask_k)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store to aa[j][i]
    aa_j_offsets = j * N + i_offsets
    final_mask = mask & mask_k
    tl.store(aa_ptr + aa_j_offsets, result, mask=final_mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    # We process i from 1 to N-1, so we have N-1 elements
    n_elements = N - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](aa, b, c, j, k, N, BLOCK_SIZE=BLOCK_SIZE)
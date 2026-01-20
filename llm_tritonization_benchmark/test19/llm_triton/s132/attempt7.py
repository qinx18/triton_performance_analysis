import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + 1
    
    mask = i_indices < N
    
    # Load c[1]
    c_val = tl.load(c_ptr + 1)
    
    # Load b[i] for valid indices
    b_vals = tl.load(b_ptr + i_indices, mask=mask, other=0.0)
    
    # Load aa[k][i-1] = aa[k * N + (i-1)]
    aa_k_indices = k * N + (i_indices - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask, other=0.0)
    
    # Compute result
    result = aa_k_vals + b_vals * c_val
    
    # Store to aa[j][i] = aa[j * N + i]
    aa_j_indices = j * N + i_indices
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    m = 0
    j = m
    k = m + 1
    
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    s132_kernel[grid](aa, b, c, j, k, N, BLOCK_SIZE=BLOCK_SIZE)
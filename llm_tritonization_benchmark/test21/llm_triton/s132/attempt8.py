import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + 1
    
    mask = i_indices < N
    
    # Load b[i]
    b_vals = tl.load(b_ptr + i_indices, mask=mask)
    
    # Load c[1] (scalar)
    c_1 = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] - need (i-1) indices
    i_minus_1 = i_indices - 1
    aa_k_indices = k * N + i_minus_1
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store to aa[j][i]
    aa_j_indices = j * N + i_indices
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # m = 0, so j = 0 and k = 1
    m = 0
    j_actual = m
    k_actual = m + 1
    
    # Grid for i from 1 to N-1 (total N-1 elements)
    num_elements = N - 1
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](aa, b, c, j_actual, k_actual, N, BLOCK_SIZE)
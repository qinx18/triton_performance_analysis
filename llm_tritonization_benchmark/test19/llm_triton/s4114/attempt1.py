import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Check bounds: i should be in range [n1-1, N)
    mask = (indices >= (n1 - 1)) & (indices < N)
    
    # Load values with mask
    b_vals = tl.load(b_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Load indirect indices k = ip[i]
    k_vals = tl.load(ip_ptr + indices, mask=mask)
    
    # Compute c indices: LEN_1D-k+1-2 = N-k+1-2 = N-k-1
    c_indices = N - k_vals - 1
    
    # Load from c array using indirect indices
    # Need to ensure c_indices are valid (0 <= c_indices < N)
    c_mask = mask & (c_indices >= 0) & (c_indices < N)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask)
    
    # Compute: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N - (n1 - 1), BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
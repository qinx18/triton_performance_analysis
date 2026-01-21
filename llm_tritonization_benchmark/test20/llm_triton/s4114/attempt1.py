import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Apply bounds: i >= n1-1 and i < N
    mask = (indices >= n1 - 1) & (indices < N)
    
    # Load ip[i] values
    k_vals = tl.load(ip_ptr + indices, mask=mask)
    
    # Load b[i] and d[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute c indices: LEN_1D - k + 1 - 2 = N - k - 1
    c_indices = N - k_vals - 1
    
    # Create mask for valid c indices
    c_mask = mask & (c_indices >= 0) & (c_indices < N)
    
    # Load c values
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute result: a[i] = b[i] + c[N-k-1] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip, n1, N, BLOCK_SIZE=BLOCK_SIZE
    )
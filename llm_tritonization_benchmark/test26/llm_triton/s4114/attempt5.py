import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices >= n1 - 1) & (indices < N)
    
    # Load ip values with mask
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate c indices: LEN_1D - k + 1 - 2 = N - k - 1
    c_indices = N - ip_vals - 1
    c_mask = mask & (c_indices >= 0) & (c_indices < N)
    
    # Load arrays
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Compute: a[i] = b[i] + c[N-k-1] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip):
    N = a.shape[0]
    n1 = 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Check bounds: i should be from (n1-1) to (N-1)
    valid_range = (indices >= (n1 - 1)) & (indices < N)
    
    # Load ip values for valid indices
    ip_vals = tl.load(ip_ptr + indices, mask=valid_range, other=0)
    
    # Load b and d values
    b_vals = tl.load(b_ptr + indices, mask=valid_range, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=valid_range, other=0.0)
    
    # Calculate c index: LEN_1D - k + 1 - 2 = N - ip_vals - 1
    c_indices = N - ip_vals - 1
    c_valid = (c_indices >= 0) & (c_indices < N) & valid_range
    c_vals = tl.load(c_ptr + c_indices, mask=c_valid, other=0.0)
    
    # Compute: a[i] = b[i] + c[N-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, result, mask=valid_range)

def s4114_triton(a, b, c, d, ip, n1):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip, n1, N, BLOCK_SIZE
    )
    
    return a
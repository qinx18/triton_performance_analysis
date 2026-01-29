import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Only process indices from n1-1 to N-1
    start_idx = n1 - 1
    valid_mask = (i_offsets >= start_idx) & (i_offsets < N)
    
    # Load ip values for valid indices
    ip_vals = tl.load(ip_ptr + i_offsets, mask=valid_mask, other=0)
    
    # Load b and d values
    b_vals = tl.load(b_ptr + i_offsets, mask=valid_mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=valid_mask, other=0.0)
    
    # Calculate c indices: LEN_1D - k + 1 - 2 = N - ip_vals - 1
    c_indices = N - ip_vals - 1
    
    # Load c values with bounds checking
    c_valid_mask = valid_mask & (c_indices >= 0) & (c_indices < N)
    c_vals = tl.load(c_ptr + c_indices, mask=c_valid_mask, other=0.0)
    
    # Compute result: a[i] = b[i] + c[N-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + i_offsets, result, mask=valid_mask)

def s4114_triton(a, b, c, d, ip):
    N = a.shape[0]
    n1 = ip.shape[0] - N + 1  # Derive n1 from the expected loop range
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a
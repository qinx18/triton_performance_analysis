import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    # Calculate starting index for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    i_vals = block_start + offsets
    
    # Apply loop bounds: i in range(n1-1, N)
    start_idx = n1 - 1
    mask = (i_vals >= start_idx) & (i_vals < N)
    
    # Load values with masking
    i_ptrs = i_vals
    b_vals = tl.load(b_ptr + i_ptrs, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_ptrs, mask=mask, other=0.0)
    
    # Load indirect indices k = ip[i]
    k_vals = tl.load(ip_ptr + i_ptrs, mask=mask, other=0)
    
    # Calculate c array indices: LEN_1D-k+1-2 = N-k-1
    c_indices = N - k_vals - 1
    
    # Clamp c_indices to valid range [0, N)
    c_indices = tl.maximum(c_indices, 0)
    c_indices = tl.minimum(c_indices, N - 1)
    
    # Load c values using indirect addressing
    c_vals = tl.load(c_ptr + c_indices, mask=mask, other=0.0)
    
    # Compute: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + i_ptrs, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
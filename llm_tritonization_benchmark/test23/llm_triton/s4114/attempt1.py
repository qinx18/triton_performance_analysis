import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Calculate actual i values: i = n1-1 + block_idx
    i_values = (n1 - 1) + indices
    mask = (i_values < N) & (indices < N)
    
    # Load ip[i] for valid indices
    ip_vals = tl.load(ip_ptr + i_values, mask=mask)
    
    # Load b[i] and d[i]
    b_vals = tl.load(b_ptr + i_values, mask=mask)
    d_vals = tl.load(d_ptr + i_values, mask=mask)
    
    # Calculate c index: LEN_1D-k+1-2 = N-k-1
    c_indices = N - ip_vals - 1
    c_vals = tl.load(c_ptr + c_indices, mask=mask)
    
    # Compute result: a[i] = b[i] + c[N-k-1] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + i_values, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    N = a.shape[0]
    start_idx = n1 - 1
    num_elements = N - start_idx
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip, n1, N, BLOCK_SIZE=BLOCK_SIZE
    )
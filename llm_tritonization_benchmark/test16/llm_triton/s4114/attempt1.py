import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the range: from (n1-1) to N-1
    start_idx = n1 - 1
    total_elements = N - start_idx
    
    # Block processing
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices in the original array
    actual_indices = start_idx + block_start + offsets
    mask = (block_start + offsets) < total_elements
    
    # Load ip values for gather operation
    ip_vals = tl.load(ip_ptr + actual_indices, mask=mask)
    
    # Load b[i] and d[i]
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    d_vals = tl.load(d_ptr + actual_indices, mask=mask)
    
    # Calculate c array indices: LEN_1D - k + 1 - 2 = N - ip_vals - 1
    c_indices = N - ip_vals - 1
    c_vals = tl.load(c_ptr + c_indices, mask=mask)
    
    # Compute a[i] = b[i] + c[N-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    N = a.shape[0]
    
    # Calculate the range: from (n1-1) to N-1
    start_idx = n1 - 1
    total_elements = N - start_idx
    
    if total_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
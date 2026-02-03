import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, N, BLOCK_SIZE: tl.constexpr):
    # Calculate starting position
    start_idx = n1 - 1
    total_elements = N - start_idx
    
    # Get block start position relative to the actual range
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Map to actual array indices
    indices = start_idx + block_start + offsets
    
    # Create mask for valid elements
    mask = (block_start + offsets) < total_elements
    
    # Load data with masking
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Compute c array indices: LEN_1D - k + 1 - 2 = N - k - 1
    c_indices = N - ip_vals - 1
    c_vals = tl.load(c_ptr + c_indices, mask=mask, other=0.0)
    
    # Compute: a[i] = b[i] + c[N-k-1] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    N = a.shape[0]
    
    # Calculate the range we need to process
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
import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n1, len_1d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from n1-1
    actual_offsets = offsets + (n1 - 1)
    mask = actual_offsets < len_1d
    
    # Load data with masking
    b_vals = tl.load(b_ptr + actual_offsets, mask=mask)
    d_vals = tl.load(d_ptr + actual_offsets, mask=mask)
    ip_vals = tl.load(ip_ptr + actual_offsets, mask=mask)
    
    # Calculate k = ip[i] and compute c array index
    k_vals = ip_vals
    c_indices = len_1d - k_vals + 1 - 2
    
    # Clamp c_indices to valid range [0, len_1d)
    c_indices = tl.maximum(0, tl.minimum(c_indices, len_1d - 1))
    
    # Load c values using computed indices
    c_vals = tl.load(c_ptr + c_indices, mask=mask)
    
    # Compute result: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + actual_offsets, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    len_1d = a.shape[0]
    
    # Calculate number of elements to process
    num_elements = len_1d - (n1 - 1)
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, len_1d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
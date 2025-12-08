import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, len_1d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices in the range [n1-1, len_1d)
    mask = (indices >= (n1 - 1)) & (indices < len_1d)
    
    # Load ip[i] values
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Load b[i] and d[i] values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Calculate c array indices: LEN_1D - k + 1 - 2 = len_1d - ip_vals - 1
    c_indices = len_1d - ip_vals - 1
    
    # Load c values using computed indices
    # Need to mask c_indices to ensure they're valid
    c_mask = mask & (c_indices >= 0) & (c_indices < len_1d)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    len_1d = a.shape[0]
    
    # Calculate number of elements to process
    n_elements = len_1d - (n1 - 1)
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_1d, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip, n1, len_1d,
        BLOCK_SIZE=BLOCK_SIZE
    )
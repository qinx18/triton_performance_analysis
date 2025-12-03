import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid indices in range [n1-1, LEN_1D)
    mask = (idx < n_elements) & (idx >= n1 - 1)
    
    # Load data with mask
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    ip_vals = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Compute c indices: LEN_1D - k + 1 - 2 = LEN_1D - ip[i] - 1
    c_idx = n_elements - ip_vals - 1
    
    # Load c values using computed indices
    # Need to ensure c_idx are valid
    c_mask = mask & (c_idx >= 0) & (c_idx < n_elements)
    c_vals = tl.load(c_ptr + c_idx, mask=c_mask, other=0.0)
    
    # Compute result: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    n_elements = a.numel()
    
    # Calculate effective range size
    range_start = n1 - 1
    range_size = n_elements - range_start
    
    if range_size <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
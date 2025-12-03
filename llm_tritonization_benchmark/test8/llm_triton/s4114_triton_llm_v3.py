import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, LEN_1D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices: i = n1-1 + offsets
    indices = (n1 - 1) + block_start + offsets
    
    # Mask for valid indices
    mask = indices < LEN_1D
    
    # Load ip[i] for indirect addressing
    ip_vals = tl.load(ip_ptr + indices, mask=mask)
    
    # Calculate c indices: LEN_1D - k + 1 - 2
    c_indices = LEN_1D - ip_vals + 1 - 2
    
    # Clamp c_indices to valid range [0, LEN_1D)
    c_indices = tl.maximum(c_indices, 0)
    c_indices = tl.minimum(c_indices, LEN_1D - 1)
    
    # Load b[i], c[LEN_1D-k+1-2], d[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + c_indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    
    # Number of elements to process: from n1-1 to LEN_1D-1
    n_elements = LEN_1D - (n1 - 1)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip, 
        n1, LEN_1D, 
        BLOCK_SIZE=BLOCK_SIZE
    )
import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, LEN_1D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE + (n1 - 1)
    
    indices = block_start + offsets
    mask = indices < LEN_1D
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + indices, mask=mask)
    
    # Load b and d values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Calculate c indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
    c_indices = LEN_1D - ip_vals - 1
    c_mask = mask & (c_indices >= 0) & (c_indices < LEN_1D)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - (n1 - 1)
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s4114_kernel[(grid_size,)](
        a, b, c, d, ip,
        n1, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )
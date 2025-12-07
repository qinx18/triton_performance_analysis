import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n1, LEN_1D,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the range of indices this block will handle
    block_start = tl.program_id(0) * BLOCK_SIZE + (n1 - 1)
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices
    mask = indices < LEN_1D
    
    # Load ip values for indirect addressing
    k_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate c array indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
    c_indices = LEN_1D - k_vals - 1
    c_mask = mask & (c_indices >= 0) & (c_indices < LEN_1D)
    
    # Load values
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Compute: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    
    # Calculate the number of elements to process
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
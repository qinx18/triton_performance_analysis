import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Pre-define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices starting from n1-1
    i_indices = (n1 - 1) + block_start + offsets
    
    # Create mask for valid indices
    mask = i_indices < LEN_1D
    
    # Load ip values for gather operation
    ip_vals = tl.load(ip_ptr + i_indices, mask=mask, other=0)
    
    # Calculate c indices: LEN_1D - k + 1 - 2 = LEN_1D - ip[i] - 1
    c_indices = LEN_1D - ip_vals - 1
    
    # Create mask for valid c indices
    c_mask = mask & (c_indices >= 0) & (c_indices < LEN_1D)
    
    # Load arrays
    b_vals = tl.load(b_ptr + i_indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    d_vals = tl.load(d_ptr + i_indices, mask=mask, other=0.0)
    
    # Compute: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + i_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    
    # Calculate number of elements to process
    n_elements = LEN_1D - (n1 - 1)
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s4114_kernel[grid](
        a, b, c, d, ip, n1, LEN_1D, BLOCK_SIZE
    )
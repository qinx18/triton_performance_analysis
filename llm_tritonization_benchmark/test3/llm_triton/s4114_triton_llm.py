import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n1, LEN_1D,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute element indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from n1-1
    actual_indices = offsets + (n1 - 1)
    
    # Mask for valid indices
    mask = actual_indices < LEN_1D
    
    # Load input values with masking
    b_vals = tl.load(b_ptr + actual_indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + actual_indices, mask=mask, other=0.0)
    ip_vals = tl.load(ip_ptr + actual_indices, mask=mask, other=0)
    
    # Compute c indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
    c_indices = LEN_1D - ip_vals - 1
    
    # Clamp c_indices to valid range to avoid out-of-bounds access
    c_indices = tl.maximum(0, tl.minimum(c_indices, LEN_1D - 1))
    
    # Load c values using computed indices
    c_vals = tl.load(c_ptr + c_indices, mask=mask, other=0.0)
    
    # Compute result: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    """
    Triton implementation of TSVC s4114
    Optimizations:
    - Vectorized computation across multiple elements per thread block
    - Coalesced memory access patterns
    - Efficient masking for boundary conditions
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    ip = ip.contiguous()
    
    LEN_1D = a.shape[0]
    
    # Calculate number of elements to process
    num_elements = LEN_1D - (n1 - 1)
    
    if num_elements <= 0:
        return a
    
    # Choose block size for optimal memory access
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
    
    # Launch kernel
    s4114_kernel[(grid_size,)](
        a, b, c, d, ip,
        n1, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a
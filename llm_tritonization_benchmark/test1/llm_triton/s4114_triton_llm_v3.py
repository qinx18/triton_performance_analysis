import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n1: tl.constexpr,
    LEN_1D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from n1-1
    i_vals = offsets + (n1 - 1)
    mask = i_vals < LEN_1D
    
    # Load ip[i] values
    ip_offsets = tl.where(mask, i_vals, 0)
    k_vals = tl.load(ip_ptr + ip_offsets, mask=mask, other=0)
    
    # Load b[i] and d[i]
    b_vals = tl.load(b_ptr + ip_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + ip_offsets, mask=mask, other=0.0)
    
    # Calculate c array indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
    c_indices = LEN_1D - k_vals - 1
    
    # Clamp c_indices to valid range to avoid out-of-bounds access
    c_indices = tl.where(c_indices >= 0, c_indices, 0)
    c_indices = tl.where(c_indices < LEN_1D, c_indices, LEN_1D - 1)
    
    # Load c values using computed indices
    c_vals = tl.load(c_ptr + c_indices, mask=mask, other=0.0)
    
    # Compute a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + ip_offsets, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    
    # Calculate number of elements to process
    num_elements = LEN_1D - (n1 - 1)
    
    if num_elements <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
    
    s4114_kernel[(grid_size,)](
        a, b, c, d, ip,
        n1=n1,
        LEN_1D=LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )
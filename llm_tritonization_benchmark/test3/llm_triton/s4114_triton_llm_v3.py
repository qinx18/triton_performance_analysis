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
    
    # Load data with masking
    b_vals = tl.load(b_ptr + i_vals, mask=mask)
    d_vals = tl.load(d_ptr + i_vals, mask=mask)
    
    # Load indirect indices
    k_vals = tl.load(ip_ptr + i_vals, mask=mask)
    
    # Calculate c array indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
    c_indices = LEN_1D - k_vals - 1
    c_vals = tl.load(c_ptr + c_indices, mask=mask)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + i_vals, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    
    # Calculate number of elements to process
    num_elements = LEN_1D - (n1 - 1)
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1=n1,
        LEN_1D=LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )
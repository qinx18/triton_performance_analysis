import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n1: tl.constexpr,
    LEN_1D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from n1-1
    actual_indices = offsets + (n1 - 1)
    
    # Create mask for valid indices
    mask = actual_indices < LEN_1D
    
    # Load data with masking
    b_vals = tl.load(b_ptr + actual_indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + actual_indices, mask=mask, other=0.0)
    
    # Load indirect indices
    k_vals = tl.load(ip_ptr + actual_indices, mask=mask, other=0)
    
    # Calculate c array indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
    c_indices = LEN_1D - k_vals - 1
    
    # Create mask for valid c indices
    c_mask = mask & (c_indices >= 0) & (c_indices < LEN_1D)
    
    # Load c values using indirect addressing
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute result: a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    
    # Calculate the range of indices to process
    num_elements = LEN_1D - (n1 - 1)
    
    if num_elements <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    
    # Launch kernel
    s4114_kernel[(num_blocks,)](
        a, b, c, d, ip,
        n1=n1,
        LEN_1D=LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
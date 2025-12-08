import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr,
    inc, j, len_2d,
    sum_ptr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, len_2d - 1)
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < (len_2d - 1)
    
    # Calculate array indices
    off_indices = inc + i_offsets
    ip_indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    aa_indices = (j - 1) * len_2d + ip_indices
    
    # Load data
    a_vals = tl.load(a_ptr + off_indices, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Sum the products for this block
    block_sum = tl.sum(products)
    
    # Atomic add to global sum
    tl.atomic_add(sum_ptr, block_sum)

def s4116_triton(a, aa, ip, inc, j):
    len_2d = aa.shape[0]
    
    # Create output tensor for sum
    sum_result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch parameters
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(len_2d - 1, BLOCK_SIZE)
    
    # Launch kernel
    s4116_kernel[(grid_size,)](
        a, aa, ip,
        inc, j, len_2d,
        sum_result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()
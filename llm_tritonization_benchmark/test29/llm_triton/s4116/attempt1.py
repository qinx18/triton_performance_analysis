import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr, sum_ptr,
    len_2d, inc, j,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Create mask for valid indices
    mask = i_offsets < (len_2d - 1)
    
    # Calculate array offsets for a
    a_indices = inc + i_offsets
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Calculate 2D indices for aa
    aa_row = j - 1
    aa_indices = aa_row * len_2d + ip_vals
    
    # Load values
    a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Mask out invalid elements
    products = tl.where(mask, products, 0.0)
    
    # Sum within block
    block_sum = tl.sum(products)
    
    # Store block sum
    tl.store(sum_ptr + pid, block_sum)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    # Calculate number of elements to process
    n_elements = len_2d - 1
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    s4116_kernel[(n_blocks,)](
        a, aa, ip, partial_sums,
        len_2d, inc, j,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum partial results
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()
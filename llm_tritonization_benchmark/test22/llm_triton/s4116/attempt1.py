import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, sum_ptr, inc, j, len_2d, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < (len_2d - 1)
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate offsets for array a (scalar expansion of off = inc + i)
    a_offsets = inc + indices
    
    # Load from array a using computed offsets
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Calculate 2D array indices for aa[j-1][ip[i]]
    aa_offsets = (j - 1) * len_2d + ip_indices
    
    # Load from 2D array aa (flattened)
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Sum the products for this block
    block_sum = tl.sum(products)
    
    # Store partial sum
    tl.store(sum_ptr + block_id, block_sum)

def s4116_triton(a, aa, ip, inc, j, len_2d):
    # Flatten 2D array aa
    aa_flat = aa.flatten()
    
    # Calculate number of blocks needed
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(len_2d - 1, BLOCK_SIZE)
    
    # Allocate output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4116_kernel[grid](
        a, aa_flat, ip, partial_sums,
        inc, j, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    result = torch.sum(partial_sums)
    
    return result
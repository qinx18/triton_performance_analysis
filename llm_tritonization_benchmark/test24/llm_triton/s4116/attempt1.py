import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, sum_ptr, inc, j, len_2d, aa_stride0, aa_stride1, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < (len_2d - 1)
    
    # Load indices for gather operation
    ip_vals = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Calculate array offsets
    off_vals = inc + i_offsets
    
    # Load a values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Calculate aa indices: aa[j-1][ip[i]]
    row_idx = j - 1
    aa_offsets = row_idx * aa_stride0 + ip_vals * aa_stride1
    
    # Load aa values (gather operation)
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Sum within block
    block_sum = tl.sum(products)
    
    # Store partial sum
    tl.store(sum_ptr + pid, block_sum)

def s4116_triton(a, aa, ip, inc, j, len_2d):
    # Calculate grid size
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(len_2d - 1, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4116_kernel[grid](
        a, aa, ip, partial_sums,
        inc, j, len_2d,
        aa.stride(0), aa.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    total_sum = torch.sum(partial_sums)
    return total_sum
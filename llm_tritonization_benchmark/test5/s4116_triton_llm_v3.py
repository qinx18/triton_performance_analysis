import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr,
    inc, j_minus_1, len_2d_minus_1, len_2d,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the range of iterations this program handles
    start_i = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = start_i + offsets
    mask = i_offsets < len_2d_minus_1
    
    # Calculate off = inc + i for each element
    off_offsets = inc + i_offsets
    
    # Load ip[i] values
    ip_vals = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_offsets, mask=mask, other=0.0)
    
    # Calculate aa[j-1][ip[i]] indices
    aa_indices = j_minus_1 * len_2d + ip_vals
    
    # Load aa[j-1][ip[i]] values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Calculate products
    products = a_vals * aa_vals
    
    # Apply mask to zero out invalid elements
    products = tl.where(mask, products, 0.0)
    
    # Sum the products for this block
    block_sum = tl.sum(products, axis=0)
    
    # Store the partial sum
    tl.store(output_ptr + pid, block_sum)

def s4116_triton(a, aa, ip, inc, j):
    len_2d = aa.shape[1]
    len_2d_minus_1 = len_2d - 1
    j_minus_1 = j - 1
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(len_2d_minus_1, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s4116_kernel[(num_blocks,)](
        a, aa, ip,
        inc, j_minus_1, len_2d_minus_1, len_2d,
        partial_sums,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum all partial results
    result = torch.sum(partial_sums)
    return result
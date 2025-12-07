import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, LEN_2D, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices for gather operation
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate offsets for array 'a'
    a_offsets = inc + indices
    a_mask = mask & (a_offsets >= 0)
    a_vals = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
    
    # Calculate 2D indices for aa array
    row_idx = j - 1
    aa_offsets = row_idx * LEN_2D + ip_vals
    aa_mask = mask & (ip_vals >= 0) & (ip_vals < LEN_2D) & (row_idx >= 0) & (row_idx < LEN_2D)
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=aa_mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Apply final mask to products
    final_mask = a_mask & aa_mask
    products = tl.where(final_mask, products, 0.0)
    
    # Sum the products
    result = tl.sum(products)
    
    # Store result (each block stores its partial sum)
    output_ptr = tl.program_id(0)
    tl.store(tl.program_id(0) + output_ptr, result)

def s4116_triton(a, aa, ip, inc, j):
    LEN_2D = aa.shape[0]
    n_elements = LEN_2D - 1
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s4116_kernel[(num_blocks,)](
        a, aa, ip,
        inc, j, LEN_2D, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum partial results
    return partial_sums.sum().item()
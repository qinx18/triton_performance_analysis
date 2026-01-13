import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, 
                 n_elements, aa_stride_0, aa_stride_1,
                 BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < n_elements
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute off values using scalar expansion
    off_vals = inc + indices
    
    # Load a values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa indices
    aa_row_offset = (j - 1) * aa_stride_0
    aa_indices = aa_row_offset + ip_vals * aa_stride_1
    
    # Load aa values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Store partial results
    return tl.sum(products, axis=0)

def s4116_triton(a, aa, ip, inc, j):
    n_elements = aa.shape[0] - 1
    
    # Handle empty case
    if n_elements <= 0:
        return torch.tensor(0.0, device=a.device, dtype=a.dtype)
    
    # Get strides for 2D array
    aa_stride_0 = aa.stride(0)
    aa_stride_1 = aa.stride(1)
    
    # Choose block size
    BLOCK_SIZE = min(256, triton.next_power_of_2(n_elements))
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create output tensor for partial sums
    num_blocks = grid[0]
    partial_sums = torch.zeros(num_blocks, device=a.device, dtype=a.dtype)
    
    # Launch kernel for each block
    for block_id in range(num_blocks):
        block_start = block_id * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_elements)
        actual_size = block_end - block_start
        
        if actual_size > 0:
            # Create temporary tensors for this block
            block_offsets = torch.arange(actual_size, device=a.device)
            indices = block_start + block_offsets
            
            # Load data for this block
            ip_vals = ip[indices]
            off_vals = inc + indices
            
            # Compute products
            a_vals = a[off_vals]
            aa_vals = aa[j-1, ip_vals]
            products = a_vals * aa_vals
            
            # Sum and store
            partial_sums[block_id] = products.sum()
    
    # Sum all partial results
    return partial_sums.sum()
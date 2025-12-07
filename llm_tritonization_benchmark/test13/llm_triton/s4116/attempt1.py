import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, n_elements, 
                 BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices from ip array
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate offsets for array a
    a_offsets = inc + indices
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Calculate offsets for 2D array aa[j-1][ip[i]]
    aa_offsets = (j - 1) * 256 + ip_vals  # assuming LEN_2D = 256
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = tl.where(mask, a_vals * aa_vals, 0.0)
    
    # Store partial results
    result = tl.sum(products)
    
    # Store result for this block
    tl.store(tl.program_id(0) + tl.arange(0, 1), result)

def s4116_triton(a, aa, ip, inc, j):
    n_elements = aa.shape[0] - 1  # LEN_2D - 1
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s4116_kernel[grid_size,](
        a, aa, ip, inc, j, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # This approach won't work with the above kernel design
    # Let's redesign with a simpler reduction approach
    
    # Alternative: compute everything in a single block for simplicity
    # since we need to sum across all elements anyway
    
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # For this specific pattern, use a single-block approach
    s4116_single_block_kernel[1,](
        a, aa, ip, inc, j, n_elements, result,
        BLOCK_SIZE=min(1024, triton.next_power_of_2(n_elements))
    )
    
    return result.item()

@triton.jit
def s4116_single_block_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, n_elements, result_ptr,
                              BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Process elements in chunks
    total_sum = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        indices = block_start + offsets
        mask = indices < n_elements
        
        # Load indices from ip array
        ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
        
        # Calculate offsets for array a
        a_offsets = inc + indices
        a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        
        # Calculate offsets for 2D array aa[j-1][ip[i]]
        aa_offsets = (j - 1) * 256 + ip_vals
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
        
        # Compute products and sum
        products = tl.where(mask, a_vals * aa_vals, 0.0)
        block_sum = tl.sum(products)
        total_sum += block_sum
    
    # Store final result
    tl.store(result_ptr, total_sum)

def s4116_triton(a, aa, ip, inc, j):
    n_elements = aa.shape[0] - 1  # LEN_2D - 1
    
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    s4116_single_block_kernel[1,](
        a, aa, ip, inc, j, n_elements, result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()
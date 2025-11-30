import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr,
    inc, j, len_2d,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel computes a reduction, so we use one block
    block_id = tl.program_id(0)
    if block_id != 0:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Process elements in blocks
    for block_start in range(0, len_2d - 1, BLOCK_SIZE):
        # Load indices for this block
        i_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = i_offsets < (len_2d - 1)
        
        # Compute offsets for array a
        a_offsets = inc + i_offsets
        
        # Load ip indices
        ip_indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
        
        # Compute aa offsets: aa[j-1][ip[i]]
        aa_offsets = (j - 1) * len_2d + ip_indices
        
        # Load from arrays
        a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
        
        # Compute products and sum
        products = a_vals * aa_vals
        sum_val += tl.sum(tl.where(mask, products, 0.0))
    
    # Store result (only one thread stores)
    if tl.program_id(0) == 0:
        result_ptr = a_ptr + 0  # Use a temporary location
        # We'll return the sum through a separate output tensor

@triton.jit
def s4116_reduction_kernel(
    a_ptr, aa_ptr, ip_ptr, output_ptr,
    inc, j, len_2d,
    BLOCK_SIZE: tl.constexpr,
):
    # Initialize sum
    sum_val = 0.0
    
    # Process all elements
    for block_start in range(0, len_2d - 1, BLOCK_SIZE):
        # Load indices for this block
        i_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = i_offsets < (len_2d - 1)
        
        # Compute offsets for array a
        a_offsets = inc + i_offsets
        
        # Load ip indices
        ip_indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
        
        # Compute aa offsets: aa[j-1][ip[i]]
        aa_offsets = (j - 1) * len_2d + ip_indices
        
        # Load from arrays
        a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
        
        # Compute products and sum
        products = a_vals * aa_vals
        sum_val += tl.sum(tl.where(mask, products, 0.0))
    
    # Store result
    tl.store(output_ptr, sum_val)

def s4116_triton(a, aa, ip, inc, j):
    len_2d = aa.shape[1]  # Assuming aa is 2D array
    
    # Create output tensor for the result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need reduction
    BLOCK_SIZE = 256
    grid = (1,)
    
    s4116_reduction_kernel[grid](
        a, aa, ip, output,
        inc, j, len_2d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.item()
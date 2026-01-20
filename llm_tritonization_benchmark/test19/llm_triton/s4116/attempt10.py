import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, result_ptr, 
                 inc, j, n_elements, aa_stride1,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid elements
    mask = indices < n_elements
    
    # Load ip indices - ensure they are valid before using
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate a array offsets: off = inc + i
    a_offsets = inc + indices
    
    # Load from a array
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Calculate aa array offsets: aa[j-1][ip[i]]
    # Ensure ip_vals are within bounds
    ip_mask = mask & (ip_vals >= 0) & (ip_vals < aa_stride1)
    aa_offsets = (j - 1) * aa_stride1 + ip_vals
    
    # Load from aa array with proper masking
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=ip_mask, other=0.0)
    
    # Compute products only where both masks are valid
    products = tl.where(ip_mask, a_vals * aa_vals, 0.0)
    
    # Store partial results
    tl.store(result_ptr + indices, products, mask=mask)

def s4116_triton(a, aa, ip, inc, j):
    # Get dimensions
    n_elements = aa.shape[0] - 1  # LEN_2D - 1
    
    # Validate inputs
    if n_elements <= 0:
        return 0.0
    
    # Create output tensor for partial results
    partial_results = torch.zeros(n_elements, dtype=torch.float32, device=a.device)
    
    # Calculate stride for aa array (number of columns)
    aa_stride1 = aa.shape[1]
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4116_kernel[grid](
        a, aa, ip, partial_results,
        inc, j, n_elements, aa_stride1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results and return as float
    total_sum = torch.sum(partial_results)
    return total_sum.item()
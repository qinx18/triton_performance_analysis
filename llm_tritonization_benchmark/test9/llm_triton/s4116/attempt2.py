import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, n_elements, aa_stride0, aa_stride1, 
                 output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Define offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = 0.0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Calculate array indices
        a_indices = inc + current_offsets
        ip_indices = current_offsets
        
        # Load ip values to get aa column indices
        ip_vals = tl.load(ip_ptr + ip_indices, mask=mask, other=0)
        
        # Load a values
        a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
        
        # Calculate aa indices: aa[j-1][ip[i]]
        aa_row = j - 1
        aa_indices = aa_row * aa_stride0 + ip_vals * aa_stride1
        
        # Load aa values
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
        
        # Compute products and accumulate
        products = a_vals * aa_vals
        masked_products = tl.where(mask, products, 0.0)
        sum_val += tl.sum(masked_products)
    
    # Atomic add to output for thread safety
    tl.atomic_add(output_ptr, sum_val)

def s4116_triton(a, aa, ip, inc, j):
    n_elements = aa.shape[0] - 1  # LEN_2D - 1
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor
    output = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    
    s4116_kernel[grid](
        a, aa, ip, inc, j, n_elements,
        aa.stride(0), aa.stride(1),
        output,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()
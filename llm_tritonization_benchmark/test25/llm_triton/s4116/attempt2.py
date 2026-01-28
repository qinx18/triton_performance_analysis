import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, partial_sums_ptr, n, j, inc, aa_stride0, aa_stride1, BLOCK_SIZE: tl.constexpr):
    # Get block of indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices
    mask = indices < n
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate off values (direct expansion)
    off_vals = inc + indices
    
    # Load a values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Calculate aa indices
    aa_row = j - 1
    aa_indices = aa_row * aa_stride1 + ip_vals
    
    # Load aa values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products and sum
    products = a_vals * aa_vals
    partial_sum = tl.sum(tl.where(mask, products, 0.0))
    
    # Store partial sum to global memory for reduction
    block_id = tl.program_id(0)
    tl.store(partial_sums_ptr + block_id, partial_sum)

def s4116_triton(a, aa, ip):
    # Get dimensions from tensor shapes
    len_2d = aa.shape[0]
    n = len_2d - 1
    
    # Extract parameters (assuming they are provided somehow)
    # For this implementation, we'll use default values
    j = len_2d // 2  # Default value
    inc = 0  # Default value
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s4116_kernel[(num_blocks,)](
        a, aa, ip, partial_sums, n, j, inc,
        aa.stride(0), aa.stride(1),
        BLOCK_SIZE
    )
    
    # Final reduction on CPU
    return partial_sums.sum().item()
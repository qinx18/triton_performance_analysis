import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate the range of indices this program will handle
    start_idx = pid * BLOCK_SIZE
    end_idx = tl.minimum(start_idx + BLOCK_SIZE, n_elements)
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = start_idx + offsets
    mask = idx < n_elements
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Load ip[i] indices
    ip_vals = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Load b[ip[i]] values using gather
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Sum within this block
    block_sum = tl.sum(products)
    
    # Store the block sum
    tl.store(output_ptr + pid, block_sum)

@triton.jit
def reduction_kernel(partial_sums_ptr, output_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    partial_vals = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    total_sum = tl.sum(partial_vals)
    
    tl.store(output_ptr, total_sum)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary storage for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch first kernel to compute partial sums
    grid = (num_blocks,)
    s4115_kernel[grid](
        a, b, ip, partial_sums,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Create final output
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch reduction kernel to sum partial results
    reduction_kernel[(1,)](
        partial_sums, output, num_blocks,
        BLOCK_SIZE=triton.next_power_of_2(num_blocks)
    )
    
    return output.item()
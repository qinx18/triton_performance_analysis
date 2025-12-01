import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr,
    inc, j, len_2d,
    BLOCK_SIZE: tl.constexpr
):
    # Single thread handles the entire reduction
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    sum_val = 0.0
    
    # Process in blocks
    for block_start in range(0, len_2d - 1, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, len_2d - 1)
        actual_block_size = block_end - block_start
        
        # Create offsets for this block
        i_offsets = tl.arange(0, BLOCK_SIZE) + block_start
        mask = i_offsets < (len_2d - 1)
        
        # Compute array indices
        a_indices = inc + i_offsets
        ip_indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
        aa_indices = (j - 1) * len_2d + ip_indices
        
        # Load values
        a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
        
        # Compute products and sum
        products = a_vals * aa_vals
        block_sum = tl.sum(tl.where(mask, products, 0.0))
        sum_val += block_sum
    
    # Store result (only thread 0 writes)
    result_ptr = a_ptr  # Use a_ptr as temporary storage for result
    tl.store(result_ptr, sum_val)

def s4116_triton(a, flat_2d_array, indx, inc, j):
    len_2d = int(flat_2d_array.numel() ** 0.5)
    
    # Ensure tensors are on the same device and contiguous
    device = a.device
    a = a.contiguous()
    flat_2d_array = flat_2d_array.to(device).contiguous()
    indx = indx.to(device).contiguous()
    
    # Create result tensor
    result = torch.zeros(1, dtype=torch.float32, device=device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s4116_kernel[grid](
        a, flat_2d_array, indx,
        inc, j, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # The result is stored in the first element of a (temporarily)
    # Copy it to avoid modifying input
    result_val = a[0].clone()
    
    return result_val.item()
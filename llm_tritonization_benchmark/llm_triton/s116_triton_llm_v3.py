import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(
    a_ptr,
    a_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_elements - 5)
    
    # Each thread processes one block of 5 elements
    base_idx = offsets * 5
    base_mask = base_idx < (n_elements - 5)
    
    # Load values from read-only copy for all required indices
    a0_vals = tl.load(a_copy_ptr + base_idx, mask=base_mask, other=0.0)
    a1_vals = tl.load(a_copy_ptr + base_idx + 1, mask=base_mask, other=0.0)
    a2_vals = tl.load(a_copy_ptr + base_idx + 2, mask=base_mask, other=0.0)
    a3_vals = tl.load(a_copy_ptr + base_idx + 3, mask=base_mask, other=0.0)
    a4_vals = tl.load(a_copy_ptr + base_idx + 4, mask=base_mask, other=0.0)
    a5_vals = tl.load(a_copy_ptr + base_idx + 5, mask=base_mask, other=0.0)
    
    # Compute results sequentially to maintain dependencies
    # a[i] = a[i + 1] * a[i]
    result0 = a1_vals * a0_vals
    
    # a[i + 1] = a[i + 2] * a[i + 1]
    result1 = a2_vals * a1_vals
    
    # a[i + 2] = a[i + 3] * a[i + 2]
    result2 = a3_vals * a2_vals
    
    # a[i + 3] = a[i + 4] * a[i + 3]
    result3 = a4_vals * a3_vals
    
    # a[i + 4] = a[i + 5] * a[i + 4]
    result4 = a5_vals * a4_vals
    
    # Store results to original array
    tl.store(a_ptr + base_idx, result0, mask=base_mask)
    tl.store(a_ptr + base_idx + 1, result1, mask=base_mask)
    tl.store(a_ptr + base_idx + 2, result2, mask=base_mask)
    tl.store(a_ptr + base_idx + 3, result3, mask=base_mask)
    tl.store(a_ptr + base_idx + 4, result4, mask=base_mask)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate number of blocks (each block processes 5 elements)
    num_blocks = (n_elements - 5 + 4) // 5  # Ceiling division
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_blocks, BLOCK_SIZE),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
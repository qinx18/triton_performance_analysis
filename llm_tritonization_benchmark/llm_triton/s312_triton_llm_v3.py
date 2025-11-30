import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Each program handles BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    
    # Compute product of elements in this block
    block_prod = tl.reduce(a_vals, axis=0, op=tl.associative_scan.MUL)
    
    # Store the partial product
    if pid == 0:
        tl.store(result_ptr + pid, block_prod)
    else:
        tl.store(result_ptr + pid, block_prod)

@triton.jit
def s312_final_reduction_kernel(partial_results_ptr, final_result_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    # Single thread reduces all partial products
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    final_prod = 1.0
    for i in range(num_blocks):
        partial_prod = tl.load(partial_results_ptr + i)
        final_prod *= partial_prod
    
    tl.store(final_result_ptr, final_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate temporary storage for partial products
    partial_results = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    final_result = torch.ones(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel to compute partial products
    grid = (num_blocks,)
    s312_kernel[grid](
        a, partial_results, n_elements, BLOCK_SIZE
    )
    
    # Launch final reduction kernel
    grid = (1,)
    s312_final_reduction_kernel[grid](
        partial_results, final_result, num_blocks, BLOCK_SIZE
    )
    
    return final_result.item()
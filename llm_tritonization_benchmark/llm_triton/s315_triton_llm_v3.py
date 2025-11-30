import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction operation to find max value and its index
    # We'll use a single block to handle the entire array for simplicity
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
        
        # Compare each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                val = tl.load(a_ptr + block_start + i)
                if val > max_val:
                    max_val = val
                    max_idx = block_start + i
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx)

@triton.jit
def s315_kernel_parallel(a_ptr, partial_max_ptr, partial_idx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data for this block
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find max in this block
    block_max = float('-inf')
    block_max_idx = -1
    
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            if vals[i] > block_max:
                block_max = vals[i]
                block_max_idx = block_start + i
    
    # Store partial results
    tl.store(partial_max_ptr + pid, block_max)
    tl.store(partial_idx_ptr + pid, block_max_idx)

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # For small arrays, use single-threaded approach
    if n_elements <= 1024:
        result = torch.zeros(2, dtype=a.dtype, device=a.device)
        
        s315_kernel[(1,)](
            a, result, n_elements, BLOCK_SIZE
        )
        
        max_val = result[0]
        max_idx = int(result[1])
        
    else:
        # For larger arrays, use parallel reduction
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        
        partial_max = torch.full((num_blocks,), float('-inf'), dtype=a.dtype, device=a.device)
        partial_idx = torch.zeros(num_blocks, dtype=torch.int32, device=a.device)
        
        s315_kernel_parallel[(num_blocks,)](
            a, partial_max, partial_idx, n_elements, BLOCK_SIZE
        )
        
        # Reduce partial results on CPU for simplicity
        partial_max_cpu = partial_max.cpu()
        partial_idx_cpu = partial_idx.cpu()
        
        max_val = float('-inf')
        max_idx = 0
        
        for i in range(num_blocks):
            if partial_max_cpu[i] > max_val:
                max_val = partial_max_cpu[i]
                max_idx = int(partial_idx_cpu[i])
    
    chksum = max_val + float(max_idx)
    
    return max_idx + max_val + 1
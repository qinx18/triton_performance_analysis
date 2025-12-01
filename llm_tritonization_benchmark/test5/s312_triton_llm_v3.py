import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize product accumulator
    prod = 1.0
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Multiply all values in the block
        block_prod = 1.0
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                block_prod *= tl.where(offsets == i, vals, 1.0).sum()
    
    # Alternative approach using reduce operations
    prod = 1.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                idx_mask = offsets == i
                val = tl.where(idx_mask, vals, 1.0)
                val_scalar = tl.sum(val)
                prod *= val_scalar
    
    # Store result
    tl.store(output_ptr, prod)

@triton.jit  
def s312_kernel_simple(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Simple sequential approach that processes one element per thread block
    pid = tl.program_id(0)
    
    if pid == 0:  # Only first program processes the reduction
        prod = 1.0
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
            
            # Multiply valid elements
            masked_vals = tl.where(mask, vals, 1.0)
            for i in range(BLOCK_SIZE):
                element_mask = offsets == i
                element_val = tl.sum(tl.where(element_mask, masked_vals, 1.0))
                if block_start + i < n_elements:
                    prod *= element_val
        
        tl.store(output_ptr, prod)

def s312_triton(a):
    n_elements = a.shape[0]
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    # Launch kernel with single program since this is a global reduction
    s312_kernel_simple[(1,)](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()
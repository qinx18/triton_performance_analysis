import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid > 0:
        return
    
    dot = 0.0
    
    # Process in chunks of 5 elements
    for i in range(0, n_elements, 5 * BLOCK_SIZE):
        # Load 5 consecutive blocks
        offsets = i + tl.arange(0, BLOCK_SIZE)
        
        # Process 5 elements at a time
        for j in range(5):
            current_offsets = offsets + j
            mask = current_offsets < n_elements
            
            if tl.sum(mask) > 0:  # Only load if there are valid elements
                a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
                b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
                dot += tl.sum(a_vals * b_vals)
    
    tl.store(output_ptr, dot)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Ensure n_elements is divisible by 5 for the unrolled loop
    if n_elements % 5 != 0:
        # Pad arrays to make them divisible by 5
        pad_size = 5 - (n_elements % 5)
        a = torch.cat([a, torch.zeros(pad_size, dtype=a.dtype, device=a.device)])
        b = torch.cat([b, torch.zeros(pad_size, dtype=b.dtype, device=b.device)])
        n_elements = a.shape[0]
    
    # Output tensor for the dot product result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block since we're computing a single dot product
    
    s352_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()
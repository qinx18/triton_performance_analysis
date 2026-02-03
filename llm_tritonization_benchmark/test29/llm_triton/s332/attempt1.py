import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any value is greater than threshold
        gt_mask = vals > t
        if tl.sum(gt_mask.to(tl.int32)) > 0:
            # Find first occurrence
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    val = tl.load(a_ptr + block_start + i)
                    if val > t:
                        tl.store(result_ptr, val)
                        tl.store(result_ptr + 1, tl.cast(block_start + i, tl.float32))
                        return

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Initialize result tensor [value, index]
    result = torch.tensor([-1.0, -2.0], device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 1024
    
    s332_kernel[(1,)](
        a, result, t, n_elements, BLOCK_SIZE
    )
    
    value = result[0].item()
    index = int(result[1].item())
    
    return value
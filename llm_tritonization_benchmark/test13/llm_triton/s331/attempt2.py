import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    j = -1
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Check which elements are < 0
        neg_mask = a_vals < 0.0
        
        # Update j for valid negative elements in this block
        for i in range(BLOCK_SIZE):
            current_idx = block_start + i
            if current_idx < n_elements:
                element_mask = (offsets == i)
                element_neg = tl.sum(tl.where(element_mask, neg_mask, False))
                if element_neg > 0:
                    j = current_idx
    
    tl.store(result_ptr, j.to(tl.int32))

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    # Launch with single program instance
    s331_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()
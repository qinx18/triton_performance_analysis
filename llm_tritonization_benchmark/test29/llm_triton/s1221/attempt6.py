import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < n_elements) & (indices >= 4)
    
    # Load a[i]
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Load b[i-4]
    prev_indices = indices - 4
    prev_mask = (prev_indices >= 0) & (indices < n_elements) & (indices >= 4)
    b_prev_vals = tl.load(b_ptr + prev_indices, mask=prev_mask)
    
    # Compute b[i] = b[i-4] + a[i]
    result = b_prev_vals + a_vals
    
    # Store result
    tl.store(b_ptr + indices, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    
    # Use strided prefix sum approach for parallelization
    stride = 4
    
    for stream in range(stride):
        # Get indices for this stream
        start_idx = stream + stride
        if start_idx >= n_elements:
            continue
            
        stream_indices = torch.arange(start_idx, n_elements, stride, device=b.device, dtype=torch.int64)
        if len(stream_indices) == 0:
            continue
        
        # Get addend values for this stream
        addend_vals = a[stream_indices]
        
        # Compute prefix sum
        prefix_sums = torch.cumsum(addend_vals, dim=0)
        
        # Add initial value and store
        b[stream_indices] = b[stream] + prefix_sums
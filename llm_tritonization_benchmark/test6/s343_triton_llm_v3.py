import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(bb_ptr, aa_ptr, flat_2d_array_ptr, len_2d: tl.constexpr):
    # This kernel processes the entire 2D arrays sequentially
    # due to the dependency on k counter
    
    k = -1
    
    for i in range(len_2d):
        for j in range(len_2d):
            # Load bb[j][i] and aa[j][i]
            bb_val = tl.load(bb_ptr + j * len_2d + i)
            aa_val = tl.load(aa_ptr + j * len_2d + i)
            
            # Check condition
            if bb_val > 0.0:
                k += 1
                # Store aa[j][i] into flat_2d_array[k]
                tl.store(flat_2d_array_ptr + k, aa_val)

def s343_triton(bb, aa):
    # Get dimensions
    len_2d = bb.shape[0]
    
    # Create output array
    flat_2d_array = torch.zeros(len_2d * len_2d, dtype=bb.dtype, device=bb.device)
    
    # Launch kernel with single thread since we need sequential processing
    # due to the k counter dependency
    grid = (1,)
    s343_kernel[grid](
        bb, aa, flat_2d_array, len_2d
    )
    
    return flat_2d_array
import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d: tl.constexpr):
    # Process elements sequentially since we need to maintain order for packing
    k = -1
    
    for i in range(len_2d):
        for j in range(len_2d):
            # Load bb[j][i] and aa[j][i]
            bb_offset = j * len_2d + i
            aa_offset = j * len_2d + i
            
            bb_val = tl.load(bb_ptr + bb_offset)
            
            if bb_val > 0.0:
                k += 1
                aa_val = tl.load(aa_ptr + aa_offset)
                tl.store(flat_2d_array_ptr + k, aa_val)

@triton.jit
def count_positives_kernel(bb_ptr, count_ptr, len_2d: tl.constexpr):
    count = 0
    for i in range(len_2d):
        for j in range(len_2d):
            bb_offset = j * len_2d + i
            bb_val = tl.load(bb_ptr + bb_offset)
            if bb_val > 0.0:
                count += 1
    tl.store(count_ptr, count)

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Reset flat_2d_array
    flat_2d_array.zero_()
    
    # First count how many positive elements we have
    count_tensor = torch.zeros(1, dtype=torch.int32, device=aa.device)
    grid = (1,)
    count_positives_kernel[grid](
        bb, count_tensor, len_2d=len_2d
    )
    
    # Launch kernel with single thread since we need sequential processing
    grid = (1,)
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        len_2d=len_2d
    )
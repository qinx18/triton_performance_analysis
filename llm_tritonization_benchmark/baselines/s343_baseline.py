import torch

def s343_pytorch(aa, bb, flat_2d_array):
    """
    PyTorch implementation of TSVC s343 - conditional array packing.
    
    Original C code:
    for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        k = -1;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                if (bb[j][i] > (real_t)0.) {
                    k++;
                    flat_2d_array[k] = aa[j][i];
                }
            }
        }
    }
    
    Args:
        aa: 2D tensor (read-only)
        bb: 2D tensor (read-only) 
        flat_2d_array: 1D tensor (read-write)
    
    Returns:
        flat_2d_array: Modified 1D tensor
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    LEN_2D = aa.shape[0]
    
    # Create mask for bb > 0
    mask = bb > 0.0
    
    # Extract elements from aa where bb > 0, column-major order (j, i indexing)
    selected_elements = aa[mask]
    
    # Update flat_2d_array with selected elements
    num_selected = selected_elements.shape[0]
    flat_2d_array[:num_selected] = selected_elements
    
    return flat_2d_array
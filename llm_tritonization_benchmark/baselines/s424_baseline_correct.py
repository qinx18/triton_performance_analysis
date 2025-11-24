import torch

def s424_pytorch(a, flat_2d_array):
    """
    PyTorch implementation of TSVC s424 function.

    Original C code:
    int vl = 63;
    xx = flat_2d_array + vl;
    for (int i = 0; i < LEN_1D - 1; i++) {
        xx[i+1] = flat_2d_array[i] + a[i];
    }

    This means xx is a pointer to flat_2d_array offset by 63.
    So: xx[i+1] = flat_2d_array[63 + i + 1] = flat_2d_array[64 + i]
    Therefore: flat_2d_array[64 + i] = flat_2d_array[i] + a[i]

    Arrays: a (r), flat_2d_array (rw)
    """
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()

    vl = 63
    LEN_1D = a.shape[0]

    # xx points to flat_2d_array + vl (pointer arithmetic)
    # Loop: for i in [0, LEN_1D-2]
    #   xx[i+1] = flat_2d_array[i] + a[i]
    #   which is: flat_2d_array[vl + i + 1] = flat_2d_array[i] + a[i]
    #   or:       flat_2d_array[64 + i] = flat_2d_array[i] + a[i]

    for i in range(LEN_1D - 1):
        flat_2d_array[vl + i + 1] = flat_2d_array[i] + a[i]

    return flat_2d_array

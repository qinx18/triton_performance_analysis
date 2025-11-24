import torch

def s423_pytorch(a, flat_2d_array):
    """
    PyTorch implementation of TSVC s423 function.

    Original C code:
    int vl = 64;
    xx = flat_2d_array + vl;
    for (int i = 0; i < LEN_1D - 1; i++) {
        flat_2d_array[i+1] = xx[i] + a[i];
    }

    This means xx is a pointer to flat_2d_array offset by 64.
    So: flat_2d_array[i+1] = flat_2d_array[64 + i] + a[i]

    Arrays: a (r), flat_2d_array (rw)
    """
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()

    vl = 64
    LEN_1D = a.shape[0]

    # xx points to flat_2d_array + vl (pointer arithmetic)
    # Loop: for i in [0, LEN_1D-2]
    #   flat_2d_array[i+1] = flat_2d_array[vl + i] + a[i]

    for i in range(LEN_1D - 1):
        flat_2d_array[i + 1] = flat_2d_array[vl + i] + a[i]

    return flat_2d_array

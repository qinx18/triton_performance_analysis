import torch

def s31111_pytorch(a):
    """
    PyTorch implementation of TSVC s31111.

    Original C code:
    // test() is a helper function that sums 4 elements:
    // real_t test(real_t* A) { return A[0] + A[1] + A[2] + A[3]; }

    for (int nl = 0; nl < 2000*iterations; nl++) {
        sum = (real_t)0.;
        sum += test(a);        // a[0:4].sum()
        sum += test(&a[4]);    // a[4:8].sum()
        sum += test(&a[8]);    // a[8:12].sum()
        sum += test(&a[12]);   // a[12:16].sum()
        sum += test(&a[16]);   // a[16:20].sum()
        sum += test(&a[20]);   // a[20:24].sum()
        sum += test(&a[24]);   // a[24:28].sum()
        sum += test(&a[28]);   // a[28:32].sum()
    }

    Args:
        a: Input tensor (read-only)

    Note: test() is a helper function defined in the C code, NOT a parameter.
    It sums 4 consecutive elements starting at the given pointer.
    """
    a = a.contiguous()

    # Inline the test() function: sums 4 elements at each offset
    # test(a) = a[0] + a[1] + a[2] + a[3]
    sum_val = 0.0
    sum_val += a[0:4].sum().item()    # test(a)
    sum_val += a[4:8].sum().item()    # test(&a[4])
    sum_val += a[8:12].sum().item()   # test(&a[8])
    sum_val += a[12:16].sum().item()  # test(&a[12])
    sum_val += a[16:20].sum().item()  # test(&a[16])
    sum_val += a[20:24].sum().item()  # test(&a[20])
    sum_val += a[24:28].sum().item()  # test(&a[24])
    sum_val += a[28:32].sum().item()  # test(&a[28])

    return sum_val

import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(
    a_ptr,
    bb_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s118 - processes all i values sequentially inside kernel,
    parallelizing over j for each i.

    Pattern copied from s1119: single kernel launch, sequential loop inside.
    """
    # Get the ID for which a[i] this thread will help compute
    pid = tl.program_id(0)

    # Each thread block handles one i value
    # We'll launch enough blocks to cover all i values, but process them sequentially
    i = pid + 1

    if i >= n:
        return

    # Process all i values sequentially (similar to s1119's row loop)
    for i_val in range(1, n):
        # Load current a[i_val] value
        a_i = tl.load(a_ptr + i_val)

        # For this i_val, accumulate: a[i] += sum(bb[j,i] * a[i-j-1] for j in 0..i-1)
        # We'll process j values in chunks
        num_j = i_val

        # Process j values in blocks
        for j_start in range(0, num_j, BLOCK_SIZE):
            j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
            j_mask = j_offsets < num_j

            # Load bb[j, i_val]
            bb_offsets = j_offsets * n + i_val
            bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)

            # Load a[i_val - j - 1]
            a_offsets = i_val - j_offsets - 1
            a_vals = tl.load(a_ptr + a_offsets, mask=j_mask, other=0.0)

            # Compute partial products
            products = bb_vals * a_vals

            # Sum and accumulate
            partial_sum = tl.sum(products, axis=0)
            a_i += partial_sum

        # Store updated a[i_val]
        tl.store(a_ptr + i_val, a_i)


def s118_triton(a, bb):
    """
    Triton implementation of s118 using s1119 pattern:
    - Single kernel launch
    - Sequential i-loop inside kernel
    - Parallel j processing for each i

    This should be faster than sequential kernel launches.
    """
    a = a.contiguous()
    bb = bb.contiguous()

    n = a.size(0)

    if n <= 1:
        return a

    BLOCK_SIZE = 128

    # Launch just ONE kernel (similar to s1119)
    # The kernel will process all i values sequentially
    grid = (1,)

    s118_kernel[grid](
        a_ptr=a,
        bb_ptr=bb,
        n=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a

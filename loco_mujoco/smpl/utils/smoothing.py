"""
Gaussian smoothing of batched 1D signals with PyTorch.
"""

import math

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    pass


def gaussian_kernel_1d(size, sigma):
    """
    Create a normalized 1D Gaussian kernel.

    Args:
        size (int): Number of elements of the kernel.
        sigma (float): Standard deviation of the Gaussian in samples.

    Returns:
        torch.Tensor: The kernel, shape ``(size,)``, centered at ``size // 2`` and summing to one.

    """
    size = int(size)
    sigma = float(sigma)
    kernel = torch.tensor([math.exp(-((x - size // 2) ** 2) / (2 * sigma ** 2)) for x in range(size)])
    kernel /= kernel.sum()
    return kernel


# Function to apply 1D Gaussian filter
def gaussian_filter_1d_batch(input_data, kernel_size, sigma, device):
    """
    Apply a 1D Gaussian filter along the last dimension of a batch of signals.

    The signals are padded by replicating their first and last values, so that the length is preserved for an odd
    kernel size.

    Args:
        input_data (torch.Tensor): Signals of shape ``(batch, 3, length)``; the filter is applied to each of the
            three channels separately.
        kernel_size (int): Number of elements of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian in samples.
        device (torch.device): Device on which the kernel is created; has to be the device of ``input_data``.

    Returns:
        torch.Tensor: The filtered signals, shape ``(batch, 3, length)`` for an odd kernel size.

    """
    # Create 1D Gaussian kernel
    kernel = gaussian_kernel_1d(kernel_size, sigma).to(device)

    # Reshape kernel for convolution
    kernel = kernel.view(1, 1, kernel_size)

    # Adjust kernel size for batch and channel
    kernel = kernel.repeat(input_data.size(1), 1, 1)

    padding_size = kernel_size // 2

    # Apply replicate padding
    padded_input = F.pad(input_data, (padding_size, padding_size), mode='replicate')

    # Apply convolution (gaussian filtering)
    filtered_data = F.conv1d(padded_input, kernel, padding="valid", groups=3)

    return filtered_data

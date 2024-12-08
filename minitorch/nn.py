from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    nh = height // kh
    nw = width // kw

    input = input.contiguous().view(batch, channel, nh, kh, nw, kw)
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    input = input.view(batch, channel, nh, nw, kh * kw)

    return input, nh, nw


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to the input tensor given kernel size."""
    batch, channel, _, _ = input.shape
    temp_tiled, nh, nw = tile(input, kernel)
    return temp_tiled.mean(dim=4).view(batch, channel, nh, nw)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor."""
    return input == max_reduce(input, dim)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for Max is max reduction."""
        dv = int(dim.item())
        ctx.save_for_backward(input, dv)
        return max_reduce(input, dv)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Max is argmax."""
        input, dv = ctx.saved_values
        return grad_output * argmax(input, dv), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction to the input tensor given dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor."""
    temp_exp = input.exp()
    temp_sum_exp = temp_exp.sum(dim=dim)
    return temp_exp / temp_sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor."""
    temp_max = max(input, dim)
    temp_exp = (input - temp_max).exp()
    temp_sum_exp = temp_exp.sum(dim=dim)
    temp_log_sum_exp = temp_sum_exp.log() + temp_max
    return input - temp_log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to the input tensor given kernel size."""
    batch, channel, _, _ = input.shape
    temp_tiled, nh, nw = tile(input, kernel)
    return max(temp_tiled, dim=4).view(batch, channel, nh, nw)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise."""
    if ignore:
        return input
    temp_mask = rand(input.shape) > p
    return input * temp_mask

from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple.

    Args:
    ----
        x (float | Tuple[float, ...]): The value to wrap.

    Returns:
    -------
        Tuple[float, ...]: The wrapped tuple.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Internal backward method to wrap the backward call.

        Args:
        ----
            ctx (Context): The context for the function.
            d_out (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the inputs.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Internal forward method to wrap the forward call.

        Args:
        ----
            ctx (Context): The context for the function.
            *inps (float): The input values.

        Returns:
        -------
            float: The result of the forward computation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given values.

        Args:
        ----
            *vals (ScalarLike): The input values.

        Returns:
        -------
            Scalar: The result of the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Compute the backward pass for the function.

        Args:
        ----
            ctx (Context): The context for the function.
            d_out (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the inputs.

        """
        return (0.0,)


class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for addition.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the addition.

        """
        return float(a + b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for addition.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for logarithm.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The input value.

        Returns:
        -------
            float: The result of the logarithm.

        """
        ctx.save_for_backward(a)
        return float(operators.log(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for logarithm.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a, b)
        return float(a * b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the inputs.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for inverse.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The input value.

        Returns:
        -------
            float: The result of the inverse.

        """
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for inverse.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negate function $f(x) = -x$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for negation.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The input value.

        Returns:
        -------
            float: The result of the negation.

        """
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for negation.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for sigmoid.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The input value.

        Returns:
        -------
            float: The result of the sigmoid.

        """
        ctx.save_for_backward(a)
        return float(operators.sigmoid(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for sigmoid.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (a,) = ctx.saved_values
        sigmoid_a = operators.sigmoid(a)
        return sigmoid_a * (1 - sigmoid_a) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for ReLU.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The input value.

        Returns:
        -------
            float: The result of the ReLU.

        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for ReLU.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for exponential.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The input value.

        Returns:
        -------
            float: The result of the exponential.

        """
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for exponential.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class Lt(ScalarFunction):
    """Less than function $f(x, y) = 1 if x < y else 0$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for less than comparison.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the comparison.

        """
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for less than comparison.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: The derivatives of the inputs.

        """
        return 0.0, 0.0


class Eq(ScalarFunction):
    """Equal function $f(x, y) = 1 if x == y else 0$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context for the function.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the comparison.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context for the function.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: The derivatives of the inputs.

        """
        return 0.0, 0.0

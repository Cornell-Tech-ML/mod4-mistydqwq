from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    Args:
    ----
        f (Any): Arbitrary function from n-scalar args to one value.
        *vals (Any): n-float values $x_0 \ldots x_{n-1}$.
        arg (int, optional): The number $i$ of the arg to compute the derivative. Defaults to 0.
        epsilon (float, optional): A small constant. Defaults to 1e-6.

    Returns:
    -------
        Any: An approximation of $f'_i(x_0, \ldots, x_{n-1})$.

    """
    temp1 = list(vals)
    temp2 = list(vals)
    temp1[arg] = temp1[arg] + epsilon
    temp2[arg] = temp2[arg] - epsilon
    return (f(*temp1) - f(*temp2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for this variable.

        Args:
        ----
            x (Any): The derivative to accumulate.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for this variable.

        Returns
        -------
            int: The unique identifier.

        """
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`).

        Returns
        -------
            bool: True if this variable is a leaf.

        """
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant.

        Returns
        -------
            bool: True if this variable is a constant.

        """
        ...

    @property
    def parents(self) -> Iterable[Variable]:
        """Parents of this variable.

        Returns
        -------
            Iterable[Variable]: The parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for this variable.

        Args:
        ----
            d_output (Any): The derivative of the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing the parent variable and its corresponding derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable (Variable): The right-most variable.

    Returns:
    -------
        Iterable[Variable]: Non-constant Variables in topological order starting from the right.

    """
    visited = {}
    res = []

    def dfs(v: Variable) -> None:
        if id(v) in visited or v.is_constant():
            return
        visited[id(v)] = True
        for p in v.parents:
            dfs(p)
        res.append(v)

    dfs(variable)
    return list(res)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Backpropagate the derivative through the computation graph."""
    topo = list(topological_sort(variable))
    derivatives = {id(variable): deriv}

    for v in reversed(topo):
        if v.is_leaf():
            v.accumulate_derivative(derivatives[id(v)])
        else:
            current_deriv = derivatives[id(v)]
            for parent, d_output in v.chain_rule(current_deriv):
                parent_id = id(parent)
                if parent_id not in derivatives:
                    derivatives[parent_id] = d_output
                else:
                    derivatives[parent_id] += d_output


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation.

        Args:
        ----
            values (Any): The values to save for backward pass.

        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values.

        Returns
        -------
            Tuple[Any, ...]: A tuple of saved values.

        """
        return self.saved_values

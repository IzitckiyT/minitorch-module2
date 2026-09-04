from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import defaultdict

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_right = vals[:arg] + (vals[arg] + epsilon,) + vals[arg+1:]
    vals_left = vals[:arg] + (vals[arg] - epsilon,) + vals[arg+1:]
    return (f(*vals_right) - f(*vals_left)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    stack_dfs = [variable]
    count_links = defaultdict(int)
    count_links[variable.unique_id] = 0
    while len(stack_dfs) > 0:
        v = stack_dfs.pop()
        if v.is_constant():
            continue
        for u in v.history.inputs:
            count_links[u.unique_id] += 1
            if count_links[u.unique_id] == 1:
                stack_dfs.append(u)
    topsort = [variable]
    stack_dfs = [variable]

    while len(stack_dfs) > 0:
        v = stack_dfs.pop()
        if v.is_constant():
            continue
        for u in v.history.inputs:
            if u.is_constant():
                continue
            count_links[u.unique_id] -= 1
            if count_links[u.unique_id] == 0:
                stack_dfs.append(u)
                topsort.append(u)
    return topsort



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    derivatives = defaultdict(float)
    derivatives[variable.unique_id] = deriv
    for v in topological_sort(variable):
        d_output = derivatives[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(d_output)
        elif not v.is_constant():
            for u, d_input in v.chain_rule(d_output):
                if not u.is_constant():
                    derivatives[u.unique_id] += d_input


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

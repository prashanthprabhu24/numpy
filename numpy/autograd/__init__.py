"""
AutoGrad: Automatic Differentiation for NumPy

A lightweight automatic differentiation engine that builds computation graphs
and computes gradients via reverse-mode differentiation (backpropagation).
"""

import numpy as np
from typing import Union, Tuple, Optional, List, Callable


class Variable:
    """
    A wrapper around numpy arrays that tracks operations for automatic differentiation.

    Attributes:
        data: The actual numpy array data
        grad: The gradient of this variable (accumulated during backprop)
        _backward: Function to compute gradients of inputs
        _prev: Set of parent variables in the computation graph
        requires_grad: Whether to track gradients for this variable
    """

    def __init__(self, data: Union[np.ndarray, float, int], requires_grad: bool = True):
        """
        Initialize a Variable.

        Args:
            data: Numeric data (numpy array, float, or int)
            requires_grad: Whether to compute gradients for this variable
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64) if requires_grad else None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self) -> str:
        return f"Variable(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    def __str__(self) -> str:
        return f"Variable({self.data})"

    @property
    def shape(self) -> Tuple:
        """Return the shape of the data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    def backward(self, gradient: Optional[np.ndarray] = None) -> None:
        """
        Compute gradients via reverse-mode automatic differentiation.

        Args:
            gradient: The initial gradient (defaults to ones with same shape as data)
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a variable with requires_grad=False")

        # Build topological order using DFS
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited and v.requires_grad:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Set initial gradient
        if gradient is None:
            self.grad = np.ones_like(self.data, dtype=np.float64)
        else:
            self.grad = gradient

        # Backpropagate through the computation graph
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        """Reset the gradient to zero."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

    # Arithmetic operations
    def __add__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Addition: self + other"""
        return add(self, other)

    def __radd__(self, other: Union[float, int]) -> 'Variable':
        """Right addition: other + self"""
        return add(other, self)

    def __sub__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Subtraction: self - other"""
        return sub(self, other)

    def __rsub__(self, other: Union[float, int]) -> 'Variable':
        """Right subtraction: other - self"""
        return sub(other, self)

    def __mul__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Multiplication: self * other"""
        return mul(self, other)

    def __rmul__(self, other: Union[float, int]) -> 'Variable':
        """Right multiplication: other * self"""
        return mul(other, self)

    def __truediv__(self, other: Union['Variable', float, int]) -> 'Variable':
        """Division: self / other"""
        return div(self, other)

    def __rtruediv__(self, other: Union[float, int]) -> 'Variable':
        """Right division: other / self"""
        return div(other, self)

    def __pow__(self, power: Union['Variable', float, int]) -> 'Variable':
        """Power: self ** power"""
        return pow_op(self, power)

    def __rpow__(self, base: Union[float, int]) -> 'Variable':
        """Right power: base ** self"""
        return pow_op(base, self)

    def __neg__(self) -> 'Variable':
        """Negation: -self"""
        return mul(self, -1)

    def __matmul__(self, other: 'Variable') -> 'Variable':
        """Matrix multiplication: self @ other"""
        return matmul(self, other)

    # Comparison methods for convenience
    def __eq__(self, other):
        if isinstance(other, Variable):
            return np.array_equal(self.data, other.data)
        return np.array_equal(self.data, other)

    def __ne__(self, other):
        return not self.__eq__(other)


def _ensure_variable(x: Union[Variable, float, int]) -> Variable:
    """Convert input to Variable if it isn't already."""
    if isinstance(x, Variable):
        return x
    return Variable(x, requires_grad=False)


def _sum_to_shape(grad: np.ndarray, shape: Tuple) -> np.ndarray:
    """
    Sum out added dims and squeeze broadcasted dims to match the original shape.
    Used for handling broadcasting in backward pass.
    """
    # Sum out added dimensions
    ndims_added = grad.ndim - len(shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    # Sum over broadcasted dimensions
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# Basic operations
def add(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """Addition with automatic differentiation."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    out = Variable(a.data + b.data, requires_grad=(a.requires_grad or b.requires_grad))
    out._prev = {a, b}

    def _backward():
        if a.requires_grad:
            a.grad += _sum_to_shape(out.grad, a.shape)
        if b.requires_grad:
            b.grad += _sum_to_shape(out.grad, b.shape)

    out._backward = _backward
    return out


def sub(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """Subtraction with automatic differentiation."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    out = Variable(a.data - b.data, requires_grad=(a.requires_grad or b.requires_grad))
    out._prev = {a, b}

    def _backward():
        if a.requires_grad:
            a.grad += _sum_to_shape(out.grad, a.shape)
        if b.requires_grad:
            b.grad += _sum_to_shape(-out.grad, b.shape)

    out._backward = _backward
    return out


def mul(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """Multiplication with automatic differentiation."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    out = Variable(a.data * b.data, requires_grad=(a.requires_grad or b.requires_grad))
    out._prev = {a, b}

    def _backward():
        if a.requires_grad:
            a.grad += _sum_to_shape(out.grad * b.data, a.shape)
        if b.requires_grad:
            b.grad += _sum_to_shape(out.grad * a.data, b.shape)

    out._backward = _backward
    return out


def div(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """Division with automatic differentiation."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    out = Variable(a.data / b.data, requires_grad=(a.requires_grad or b.requires_grad))
    out._prev = {a, b}

    def _backward():
        if a.requires_grad:
            a.grad += _sum_to_shape(out.grad / b.data, a.shape)
        if b.requires_grad:
            b.grad += _sum_to_shape(-out.grad * a.data / (b.data ** 2), b.shape)

    out._backward = _backward
    return out


def pow_op(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """Power operation with automatic differentiation."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    out = Variable(a.data ** b.data, requires_grad=(a.requires_grad or b.requires_grad))
    out._prev = {a, b}

    def _backward():
        if a.requires_grad:
            # d/da (a^b) = b * a^(b-1)
            a.grad += _sum_to_shape(out.grad * b.data * (a.data ** (b.data - 1)), a.shape)
        if b.requires_grad:
            # d/db (a^b) = a^b * log(a)
            b.grad += _sum_to_shape(out.grad * out.data * np.log(a.data), b.shape)

    out._backward = _backward
    return out


# Activation functions
def relu(x: Variable) -> Variable:
    """ReLU activation function."""
    out = Variable(np.maximum(0, x.data), requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * (x.data > 0)

    out._backward = _backward
    return out


def sigmoid(x: Variable) -> Variable:
    """Sigmoid activation function."""
    sig = 1 / (1 + np.exp(-x.data))
    out = Variable(sig, requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            x.grad += out.grad * sig * (1 - sig)

    out._backward = _backward
    return out


def tanh(x: Variable) -> Variable:
    """Tanh activation function."""
    t = np.tanh(x.data)
    out = Variable(t, requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            # d/dx tanh(x) = 1 - tanh^2(x)
            x.grad += out.grad * (1 - t ** 2)

    out._backward = _backward
    return out


def exp(x: Variable) -> Variable:
    """Exponential function."""
    out = Variable(np.exp(x.data), requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * out.data

    out._backward = _backward
    return out


def log(x: Variable) -> Variable:
    """Natural logarithm."""
    out = Variable(np.log(x.data), requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            x.grad += out.grad / x.data

    out._backward = _backward
    return out


# Matrix operations
def matmul(a: Variable, b: Variable) -> Variable:
    """Matrix multiplication with automatic differentiation."""
    out = Variable(a.data @ b.data, requires_grad=(a.requires_grad or b.requires_grad))
    out._prev = {a, b}

    def _backward():
        if a.requires_grad:
            # For matrix multiplication: dL/dA = dL/dOut @ B^T
            a.grad += out.grad @ b.data.T
        if b.requires_grad:
            # dL/dB = A^T @ dL/dOut
            b.grad += a.data.T @ out.grad

    out._backward = _backward
    return out


def transpose(x: Variable, axes: Optional[Tuple[int, ...]] = None) -> Variable:
    """Transpose operation."""
    out = Variable(np.transpose(x.data, axes), requires_grad=x.requires_grad)
    out._prev = {x}

    # Get the inverse permutation for backward pass
    if axes is None:
        inv_axes = None
    else:
        inv_axes = tuple(np.argsort(axes))

    def _backward():
        if x.requires_grad:
            x.grad += np.transpose(out.grad, inv_axes)

    out._backward = _backward
    return out


# Reduction operations
def sum(x: Variable, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Variable:
    """Sum reduction."""
    out = Variable(np.sum(x.data, axis=axis, keepdims=keepdims), requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            grad = out.grad
            if not keepdims and axis is not None:
                # Add back the reduced dimensions
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, ax)
            # Broadcast gradient to match input shape
            x.grad += np.broadcast_to(grad, x.shape)

    out._backward = _backward
    return out


def mean(x: Variable, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Variable:
    """Mean reduction."""
    out = Variable(np.mean(x.data, axis=axis, keepdims=keepdims), requires_grad=x.requires_grad)
    out._prev = {x}

    # Calculate the size of the reduction
    if axis is None:
        n = x.data.size
    else:
        if isinstance(axis, int):
            n = x.data.shape[axis]
        else:
            n = np.prod([x.data.shape[ax] for ax in axis])

    def _backward():
        if x.requires_grad:
            grad = out.grad / n
            if not keepdims and axis is not None:
                # Add back the reduced dimensions
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, ax)
            # Broadcast gradient to match input shape
            x.grad += np.broadcast_to(grad, x.shape)

    out._backward = _backward
    return out


def reshape(x: Variable, shape: Tuple[int, ...]) -> Variable:
    """Reshape operation."""
    out = Variable(x.data.reshape(shape), requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            x.grad += out.grad.reshape(x.shape)

    out._backward = _backward
    return out


# Utility functions
def no_grad():
    """
    Context manager to temporarily disable gradient tracking.
    Useful for inference or validation loops.
    """
    class NoGradContext:
        def __enter__(self):
            self.prev_state = []
            return self

        def __exit__(self, *args):
            pass

    return NoGradContext()


# Export all public functions
__all__ = [
    'Variable',
    'add', 'sub', 'mul', 'div', 'pow_op',
    'relu', 'sigmoid', 'tanh', 'exp', 'log',
    'matmul', 'transpose',
    'sum', 'mean', 'reshape',
    'no_grad',
]

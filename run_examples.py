"""
Examples demonstrating the AutoGrad functionality.
"""

import numpy as np
from numpy.autograd import Variable, relu, sigmoid, tanh, exp, log, matmul, sum, mean


def example_basic_operations():
    """Example: Basic arithmetic operations with gradient computation."""
    print("=" * 60)
    print("Example 1: Basic Operations")
    print("=" * 60)

    # Create variables
    x = Variable(3.0)
    y = Variable(4.0)

    # Compute: z = (x + y) * (x - y)
    z = (x + y) * (x - y)

    print(f"x = {x.data}, y = {y.data}")
    print(f"z = (x + y) * (x - y) = {z.data}")

    # Compute gradients
    z.backward()

    print(f"dz/dx = {x.grad}")  # Should be 2*x = 6
    print(f"dz/dy = {y.grad}")  # Should be 2*y = 8
    print()


def example_power_and_division():
    """Example: Power and division operations."""
    print("=" * 60)
    print("Example 2: Power and Division")
    print("=" * 60)

    x = Variable(2.0)
    y = Variable(3.0)

    # z = x^2 / y
    z = x ** 2 / y

    print(f"x = {x.data}, y = {y.data}")
    print(f"z = x^2 / y = {z.data}")

    z.backward()

    print(f"dz/dx = {x.grad}")  # Should be 2*x/y = 4/3
    print(f"dz/dy = {y.grad}")  # Should be -x^2/y^2 = -4/9
    print()


def example_activation_functions():
    """Example: Activation functions."""
    print("=" * 60)
    print("Example 3: Activation Functions")
    print("=" * 60)

    x = Variable(np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]))

    # ReLU
    y_relu = relu(x)
    print("ReLU:")
    print(f"Input: {x.data}")
    print(f"Output: {y_relu.data}")

    loss_relu = sum(y_relu)
    loss_relu.backward()
    print(f"Gradient: {x.grad}")
    print()

    # Reset gradient
    x.zero_grad()

    # Sigmoid
    y_sigmoid = sigmoid(x)
    print("Sigmoid:")
    print(f"Input: {x.data}")
    print(f"Output: {y_sigmoid.data}")

    loss_sigmoid = sum(y_sigmoid)
    loss_sigmoid.backward()
    print(f"Gradient: {x.grad}")
    print()

    # Reset gradient
    x.zero_grad()

    # Tanh
    y_tanh = tanh(x)
    print("Tanh:")
    print(f"Input: {x.data}")
    print(f"Output: {y_tanh.data}")

    loss_tanh = sum(y_tanh)
    loss_tanh.backward()
    print(f"Gradient: {x.grad}")
    print()


def example_matrix_multiplication():
    """Example: Matrix multiplication and linear layer."""
    print("=" * 60)
    print("Example 4: Matrix Multiplication (Linear Layer)")
    print("=" * 60)

    # Simulate a simple linear layer: y = X @ W + b
    X = Variable(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # (3, 2)
    W = Variable(np.array([[0.5, 0.3], [0.2, 0.4]]))  # (2, 2)
    b = Variable(np.array([0.1, 0.2]))  # (2,)

    # Forward pass
    y = X @ W + b

    print(f"X shape: {X.shape}, W shape: {W.shape}, b shape: {b.shape}")
    print(f"y = X @ W + b:")
    print(y.data)

    # Compute loss (mean squared)
    loss = mean(y ** 2)
    print(f"\nLoss (mean of y^2): {loss.data}")

    # Backward pass
    loss.backward()

    print(f"\nGradients:")
    print(f"dL/dX:\n{X.grad}")
    print(f"dL/dW:\n{W.grad}")
    print(f"dL/db: {b.grad}")
    print()


def example_neural_network():
    """Example: Simple 2-layer neural network."""
    print("=" * 60)
    print("Example 5: Simple Neural Network")
    print("=" * 60)

    # Input
    X = Variable(np.array([[1.0, 2.0]]))  # (1, 2)

    # Layer 1: 2 -> 3
    W1 = Variable(np.random.randn(2, 3) * 0.5)
    b1 = Variable(np.zeros(3))

    # Layer 2: 3 -> 1
    W2 = Variable(np.random.randn(3, 1) * 0.5)
    b2 = Variable(np.zeros(1))

    # Forward pass
    h1 = X @ W1 + b1
    h1_act = relu(h1)
    output = h1_act @ W2 + b2

    print(f"Input: {X.data}")
    print(f"Hidden layer: {h1_act.data}")
    print(f"Output: {output.data}")

    # Loss (simple squared output)
    loss = output ** 2

    print(f"Loss: {loss.data}")

    # Backward pass
    loss.backward()

    print(f"\nGradients:")
    print(f"dL/dW1:\n{W1.grad}")
    print(f"dL/db1: {b1.grad}")
    print(f"dL/dW2:\n{W2.grad}")
    print(f"dL/db2: {b2.grad}")
    print()


def example_broadcasting():
    """Example: Broadcasting in operations."""
    print("=" * 60)
    print("Example 6: Broadcasting")
    print("=" * 60)

    # Vector + scalar
    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = Variable(5.0)

    z = x + y
    print(f"x = {x.data}, y = {y.data}")
    print(f"x + y = {z.data}")

    loss = sum(z)
    loss.backward()

    print(f"dL/dx = {x.grad}")
    print(f"dL/dy = {y.grad}")  # Should accumulate gradients from all elements
    print()


def example_gradient_descent():
    """Example: Simple gradient descent optimization."""
    print("=" * 60)
    print("Example 7: Gradient Descent Optimization")
    print("=" * 60)

    # Target: minimize f(x) = (x - 3)^2
    x = Variable(0.0)
    learning_rate = 0.1

    print("Optimizing f(x) = (x - 3)^2")
    print(f"Initial x: {x.data}")

    for i in range(20):
        # Forward pass
        loss = (x - 3.0) ** 2

        # Backward pass
        x.zero_grad()
        loss.backward()

        # Update
        x.data -= learning_rate * x.grad

        if i % 5 == 0:
            print(f"Step {i}: x = {x.data:.4f}, loss = {loss.data:.4f}, grad = {x.grad:.4f}")

    print(f"Final x: {x.data:.4f} (target: 3.0)")
    print()


def example_computational_graph():
    """Example: Complex computational graph."""
    print("=" * 60)
    print("Example 8: Complex Computational Graph")
    print("=" * 60)

    # Create a complex expression
    a = Variable(2.0)
    b = Variable(3.0)
    c = Variable(4.0)

    # f = (a * b + c) * (a - c) + b^2
    d = a * b + c
    e = a - c
    f = d * e + b ** 2

    print(f"a = {a.data}, b = {b.data}, c = {c.data}")
    print(f"f = (a * b + c) * (a - c) + b^2 = {f.data}")

    f.backward()

    print(f"df/da = {a.grad}")
    print(f"df/db = {b.grad}")
    print(f"df/dc = {c.grad}")
    print()


def main():
    """Run all examples."""
    np.random.seed(42)

    example_basic_operations()
    example_power_and_division()
    example_activation_functions()
    example_matrix_multiplication()
    example_neural_network()
    example_broadcasting()
    example_gradient_descent()
    example_computational_graph()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

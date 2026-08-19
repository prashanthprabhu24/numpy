<h1 align="center">
<img src="https://raw.githubusercontent.com/numpy/numpy/main/branding/logo/primary/numpylogo.svg" width="300">
</h1><br>

# AutoGrad for NumPy

A lightweight automatic differentiation engine built on NumPy for educational purposes and experiments. This implementation provides PyTorch-like automatic differentiation capabilities using pure NumPy.

## Features

- **Automatic Differentiation**: Reverse-mode automatic differentiation (backpropagation)
- **Computation Graph**: Automatically builds and traverses computation graphs
- **NumPy Integration**: Seamless integration with NumPy arrays
- **Broadcasting Support**: Handles NumPy broadcasting in gradient computation
- **Rich Operations**: Arithmetic, activation functions, matrix operations, and reductions
- **Natural Syntax**: Pythonic operator overloading (`+`, `-`, `*`, `/`, `**`, `@`)

## Installation & Setup

This is a development version of NumPy with AutoGrad added. The autograd module is located at `numpy/autograd/`.

### Quick Start

1. Clone this repository
2. Run the examples:
   ```bash
   python run_examples.py
   ```

## Usage Examples

### Example 1: Basic Gradient Computation

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'numpy'))

from numpy.autograd import Variable
import numpy as np

# Create variables
x = Variable(3.0)
y = Variable(4.0)

# Build computation graph
z = (x + y) * (x - y)  # z = 7 * (-1) = -7

# Compute gradients
z.backward()

print(f"z = {z.data}")        # -7.0
print(f"dz/dx = {x.grad}")    # 2*x = 6.0
print(f"dz/dy = {y.grad}")    # 2*y = 8.0
```

### Example 2: Matrix Operations & Linear Layers

```python
from numpy.autograd import Variable, mean
import numpy as np

# Create matrices
X = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))  # (2, 2)
W = Variable(np.array([[0.5, 0.3], [0.2, 0.4]]))  # (2, 2)
b = Variable(np.array([0.1, 0.2]))                # (2,)

# Linear transformation: y = X @ W + b
y = X @ W + b

# Compute loss (mean squared)
loss = mean(y ** 2)
loss.backward()

print(f"Loss: {loss.data}")
print(f"∇W:\n{W.grad}")
print(f"∇b: {b.grad}")
```

### Example 3: Neural Network with Activation Functions

```python
from numpy.autograd import Variable, relu, sigmoid
import numpy as np

np.random.seed(42)

# Input
X = Variable(np.array([[1.0, 2.0]]))  # (1, 2)

# Layer 1: 2 -> 4
W1 = Variable(np.random.randn(2, 4) * 0.5)
b1 = Variable(np.zeros(4))

# Layer 2: 4 -> 1
W2 = Variable(np.random.randn(4, 1) * 0.5)
b2 = Variable(np.zeros(1))

# Forward pass
h1 = relu(X @ W1 + b1)      # Hidden layer with ReLU
output = sigmoid(h1 @ W2 + b2)  # Output with Sigmoid

# Compute loss
loss = output ** 2
loss.backward()

print(f"Output: {output.data}")
print(f"Loss: {loss.data}")
print(f"Gradients computed for all parameters!")
```

### Example 4: Training Loop (XOR Problem)

```python
from numpy.autograd import Variable, sigmoid, mean
import numpy as np

# XOR dataset
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Initialize network (2 -> 4 -> 1)
np.random.seed(42)
W1 = Variable(np.random.randn(2, 4) * 0.5)
b1 = Variable(np.zeros(4))
W2 = Variable(np.random.randn(4, 1) * 0.5)
b2 = Variable(np.zeros(1))

learning_rate = 0.5
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    X = Variable(X_data, requires_grad=False)
    y_true = Variable(y_data, requires_grad=False)

    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)

    # Loss (MSE)
    loss = mean((y_pred - y_true) ** 2)

    # Backward pass
    W1.zero_grad()
    b1.zero_grad()
    W2.zero_grad()
    b2.zero_grad()
    loss.backward()

    # Gradient descent update
    W1.data -= learning_rate * W1.grad
    b1.data -= learning_rate * b1.grad
    W2.data -= learning_rate * W2.grad
    b2.data -= learning_rate * b2.grad

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.data:.6f}")

# Test the trained network
X_test = Variable(X_data, requires_grad=False)
predictions = sigmoid(sigmoid(X_test @ W1 + b1) @ W2 + b2)
print("\nPredictions:")
for inp, pred, true in zip(X_data, predictions.data, y_data):
    print(f"{inp} -> {pred[0]:.4f} (true: {true[0]})")
```

### Example 5: Physics - Velocity and Acceleration

```python
from numpy.autograd import Variable

# Position function: s(t) = 5t² + 2t + 1
def position(t):
    return 5 * t**2 + 2 * t + 1

# Compute derivatives at t=3
t = Variable(3.0)
s = position(t)

print(f"Position at t=3: {s.data} meters")

# Velocity is ds/dt
s.backward()
print(f"Velocity at t=3: {t.grad} m/s")  # 10*3 + 2 = 32 m/s

# For acceleration (second derivative)
t2 = Variable(3.0)
v = 10 * t2 + 2  # velocity function
v.backward()
print(f"Acceleration: {t2.grad} m/s²")  # 10 m/s²
```

### Example 6: Optimization (Finding Minima)

```python
from numpy.autograd import Variable

# Minimize f(x) = (x - 5)²
x = Variable(0.0)
learning_rate = 0.1

print("Finding minimum of f(x) = (x - 5)²")
for i in range(20):
    loss = (x - 5.0) ** 2

    x.zero_grad()
    loss.backward()

    # Gradient descent step
    x.data -= learning_rate * x.grad

    if i % 5 == 0:
        print(f"Step {i}: x = {x.data:.4f}, f(x) = {loss.data:.4f}")

print(f"\nMinimum found at x = {x.data:.4f} (true minimum: 5.0)")
```

### Example 7: Linear Regression

```python
from numpy.autograd import Variable, mean
import numpy as np

# Generate data: y = 3x + 2 + noise
np.random.seed(42)
X_data = np.random.randn(100, 1)
y_data = 3 * X_data + 2 + np.random.randn(100, 1) * 0.5

# Initialize parameters
w = Variable(0.0)
b = Variable(0.0)

learning_rate = 0.01

print("Training Linear Regression...")
for epoch in range(100):
    X = Variable(X_data, requires_grad=False)
    y_true = Variable(y_data, requires_grad=False)

    # Prediction
    y_pred = X * w + b

    # MSE Loss
    loss = mean((y_pred - y_true) ** 2)

    # Backward
    w.zero_grad()
    b.zero_grad()
    loss.backward()

    # Update
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss.data:.4f}, w={w.data:.4f}, b={b.data:.4f}")

print(f"\nLearned: w={w.data:.4f}, b={b.data:.4f}")
print(f"True: w=3.0, b=2.0")
```

### Example 8: Multi-Variable Function

```python
from numpy.autograd import Variable, exp

# Function: f(x, y) = x² + xy + exp(y)
x = Variable(2.0)
y = Variable(1.0)

z = x**2 + x * y + exp(y)

print(f"f(2, 1) = {z.data}")

z.backward()

print(f"∂f/∂x = {x.grad}")  # 2x + y = 5
print(f"∂f/∂y = {y.grad}")  # x + exp(y) ≈ 4.718
```

### Example 9: Batch Operations

```python
from numpy.autograd import Variable, sigmoid, sum
import numpy as np

# Batch of data
X = Variable(np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
]))  # (3, 2) - 3 samples, 2 features

W = Variable(np.array([[0.5], [0.3]]))  # (2, 1)
b = Variable(0.1)

# Forward pass for all samples
y = sigmoid(X @ W + b)  # (3, 1)

loss = sum(y)
loss.backward()

print(f"Outputs:\n{y.data}")
print(f"\n∇W:\n{W.grad}")
print(f"∇b: {b.grad}")
```

### Example 10: Custom Loss Function

```python
from numpy.autograd import Variable, mean, log
import numpy as np

# Binary cross-entropy loss
def binary_cross_entropy(y_pred, y_true):
    """BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))"""
    epsilon = Variable(1e-7, requires_grad=False)  # For numerical stability
    loss = -mean(
        y_true * log(y_pred + epsilon) +
        (1 - y_true) * log(1 - y_pred + epsilon)
    )
    return loss

# Example usage
y_pred = Variable(np.array([0.9, 0.1, 0.8, 0.3]))
y_true = Variable(np.array([1.0, 0.0, 1.0, 0.0]), requires_grad=False)

loss = binary_cross_entropy(y_pred, y_true)
loss.backward()

print(f"BCE Loss: {loss.data}")
print(f"∇y_pred: {y_pred.grad}")
```

## Running Examples

### Full Examples Suite
```bash
python run_examples.py
```

This runs 8 comprehensive examples:
1. Basic arithmetic operations
2. Power and division operations
3. Activation functions (ReLU, Sigmoid, Tanh)
4. Matrix multiplication and linear layers
5. Simple neural network (2 layers)
6. Gradient descent optimization
7. Physics application (velocity & acceleration)
8. Training with backpropagation (XOR problem)

## Supported Operations

### Arithmetic Operations
- `a + b` - Addition
- `a - b` - Subtraction
- `a * b` - Multiplication
- `a / b` - Division
- `a ** b` - Power
- `-a` - Negation

### Activation Functions
- `relu(x)` - Rectified Linear Unit
- `sigmoid(x)` - Sigmoid function
- `tanh(x)` - Hyperbolic tangent
- `exp(x)` - Exponential
- `log(x)` - Natural logarithm

### Matrix Operations
- `a @ b` or `matmul(a, b)` - Matrix multiplication
- `transpose(x, axes)` - Transpose with optional axes permutation
- `reshape(x, shape)` - Reshape operation

### Reduction Operations
- `sum(x, axis, keepdims)` - Sum reduction
- `mean(x, axis, keepdims)` - Mean reduction

## API Reference

### Variable Class

```python
Variable(data, requires_grad=True)
```

**Parameters:**
- `data`: numpy array, float, or int
- `requires_grad`: bool, whether to track gradients (default: True)

**Methods:**
- `.backward(gradient=None)` - Compute gradients via backpropagation
- `.zero_grad()` - Reset gradients to zero

**Properties:**
- `.data` - NumPy array containing the values
- `.grad` - NumPy array containing the gradients
- `.shape` - Shape of the data
- `.ndim` - Number of dimensions
- `.requires_grad` - Whether gradients are tracked

## How It Works

The AutoGrad engine uses **reverse-mode automatic differentiation**:

1. **Forward Pass**: Operations on `Variable` objects build a computation graph
   - Each operation creates a new Variable
   - Parent variables are tracked in `_prev`
   - Backward function is stored in `_backward`

2. **Backward Pass**: Calling `.backward()` triggers gradient computation
   - Topological sort ensures correct order
   - Chain rule applied automatically
   - Gradients accumulated in `.grad`

3. **Gradient Flow**: Each operation knows how to compute parent gradients
   - Example: For `z = x * y`, we have:
     - `∂z/∂x = y`
     - `∂z/∂y = x`

### Example of Computation Graph

```python
x = Variable(2.0)
y = Variable(3.0)
z = x * y + x  # z = 2*3 + 2 = 8

# Graph: x ─┬─> mul ─┬─> add ─> z
#           │         │
#           └────────>┘
#        y ─────>
```

When `z.backward()` is called:
1. Start from `z` with gradient = 1.0
2. Propagate to `add`: gradients = 1.0 for both inputs
3. Propagate to `mul`: `∂mul/∂x = y = 3`, `∂mul/∂y = x = 2`
4. Accumulate at `x`: `grad = 3 + 1 = 4` (from mul and add)
5. Accumulate at `y`: `grad = 2` (from mul)

## Import Patterns

### For Scripts in Project Root

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'numpy'))

from autograd import Variable, relu, sigmoid
import numpy as np  # System NumPy for array operations
```

### Using Provided Scripts (Recommended)

```python
# Just run the provided scripts
python run_examples.py
python test_autograd.py
```

## Real-World Applications

### Physics
- Computing derivatives for motion equations
- Velocity and acceleration calculations
- Optimization of physical systems

### Machine Learning
- Training neural networks
- Backpropagation
- Gradient descent optimization
- Custom loss functions

### Economics
- Marginal analysis
- Profit optimization
- Elasticity calculations

### Engineering
- Sensitivity analysis
- Parameter optimization
- Control systems

## Educational Purpose

This implementation is designed for **learning and experimentation**.

### For Production Use:
- **PyTorch** - Full-featured deep learning framework with GPU support
- **JAX** - High-performance autodiff with XLA compilation
- **TensorFlow** - Comprehensive ML platform
- **Autograd** (HIPS) - Mature NumPy autodiff library

### Why Use This?
- ✅ Understand how autodiff works internally
- ✅ Learn backpropagation from scratch
- ✅ Experiment with gradient-based optimization
- ✅ Educational projects and demonstrations
- ✅ Lightweight with no heavy dependencies

## Architecture

```
numpy/
└── autograd/
    └── __init__.py      # Complete autodiff implementation
        ├── Variable     # Core class for gradient tracking
        ├── Operations   # add, sub, mul, div, pow
        ├── Activations  # relu, sigmoid, tanh, exp, log
        ├── Matrix Ops   # matmul, transpose
        └── Reductions   # sum, mean, reshape
```

## Contributing

This is an experimental addition to NumPy for educational purposes. Feel free to:
- Add more operations (conv, pooling, etc.)
- Improve gradient computation efficiency
- Add more examples and tutorials
- Report bugs or suggest improvements

## License

This project extends NumPy for educational experiments in automatic differentiation.

## Acknowledgments

Inspired by:
- **PyTorch** - API design and operator overloading
- **micrograd** (Andrej Karpathy) - Minimalist autodiff engine
- **Autograd** (HIPS) - NumPy-based autodiff

---

<p align="center">
Built with ❤️ for learning automatic differentiation
</p>

import torch
import matplotlib.pyplot as plt

#%%
import matplotlib.pyplot as plt
# Create a tensor that tracks gradients
x = torch.tensor(1.0, requires_grad=True)

# Define the equation
y = (x - 3) * (x - 6) * (x - 4)

# Compute the gradient
y.backward()
print("Gradient at x=1:", x.grad.item())

# Now plot the function
import numpy as np

X = np.linspace(0, 10, 100)
Y = (X - 3) * (X - 6) * (X - 4)

plt.plot(X, Y, label='y = (x-3)(x-6)(x-4)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Function with Gradient Example')

# Draw the tangent line at x=1
x_point = 1
y_point = (x_point - 3)*(x_point - 6)*(x_point - 4)
slope = 31

# tangent line equation: y = m(x - x0) + y0
tangent = slope * (X - x_point) + y_point
plt.plot(X, tangent, 'r--', label='Tangent line (slope=31)')

plt.legend()
plt.grid(True)
plt.show()
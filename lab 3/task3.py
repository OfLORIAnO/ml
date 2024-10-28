import numpy as np

np.random.seed(3)
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]

# Training data for XOR
x_train = np.array(
    [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]
)

y_train = np.array([0.0, 1.0, 1.0, 0.0])


# Function to initialize weights
def neuron_w(input_count):
    weights = np.zeros(input_count + 1)
    for i in range(1, input_count + 1):
        weights[i] = np.random.uniform(-1.0, 1.0)
    return weights


# Initialize weights for each neuron
n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]
n_y = [0, 0, 0]
n_error = [0, 0, 0]


# Function to show current weights
def show_learning():
    print("Current weights:")
    for i, w in enumerate(n_w):
        print(f"neuron {i}: w0 = {w[0]:.2f}, w1 = {w[1]:.2f}, w2 = {w[2]:.2f}")
    print("----------------")


# Forward pass function
def forward_pass(x):
    global n_y
    n_y[0] = np.tanh(np.dot(n_w[0], x))
    n_y[1] = np.tanh(np.dot(n_w[1], x))
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))


# Backward pass function
def backward_pass(y_truth):
    global n_error
    error_prime = y_truth - n_y[2]
    derivative = n_y[2] * (1.0 - n_y[2])
    n_error[2] = error_prime * derivative
    derivative = 1.0 - n_y[0] ** 2
    n_error[0] = n_w[2][1] * n_error[2] * derivative
    derivative = 1.0 - n_y[1] ** 2
    n_error[1] = n_w[2][2] * n_error[2] * derivative


# Adjust weights function
def adjust_weights(x):
    global n_w
    n_w[0] += x * LEARNING_RATE * n_error[0]
    n_w[1] += x * LEARNING_RATE * n_error[1]
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    n_w[2] += n2_inputs * LEARNING_RATE * n_error[2]


# Training loop
all_correct = False
while not all_correct:  # Train until converged
    all_correct = True
    np.random.shuffle(index_list)  # Randomize order
    for i in index_list:  # Train on all examples
        forward_pass(x_train[i])
        backward_pass(y_train[i])
        adjust_weights(x_train[i])
        show_learning()  # Show updated weights
        for i in range(len(x_train)):  # Check if converged
            forward_pass(x_train[i])
            print(
                f"x1 = {x_train[i][1]:.1f}, x2 = {x_train[i][2]:.1f}, y = {n_y[2]:.4f}"
            )
            if ((y_train[i] < 0.5) and (n_y[2] >= 0.5)) or (
                (y_train[i] >= 0.5) and (n_y[2] < 0.5)
            ):
                all_correct = False

import numpy as np


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((out_channels, 1))
        self.last_input = None

    def forward(self, input):
        self.last_input = input
        in_channels, H, W = input.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1

        output = np.zeros((self.out_channels, H_out, W_out))

        for oc in range(self.out_channels):
            for i in range(H_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                for j in range(W_out):
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    patch = input[:, h_start:h_end, w_start:w_end]
                    output[oc, i, j] = np.sum(patch * self.weights[oc]) + self.biases[oc]
        return output

    def backward(self, d_out, learning_rate):
        input = self.last_input
        _, H, W = input.shape
        H_out, W_out = d_out.shape[1], d_out.shape[2]

        d_input = np.zeros_like(input)
        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)

        for oc in range(self.out_channels):
            d_bias = 0.0  # Accumulate bias gradient for this channel
            for i in range(H_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                for j in range(W_out):
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    patch = input[:, h_start:h_end, w_start:w_end]
                    grad = d_out[oc, i, j]

                    # Update gradients
                    d_weights[oc] += grad * patch
                    d_input[:, h_start:h_end, w_start:w_end] += grad * self.weights[oc]
                    d_bias += grad

            d_biases[oc] = d_bias  # Set accumulated bias gradient

        # Update parameters
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input


class ReLULayer:
    def __init__(self):
        self.last_input = None

    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, d_out):
        d_input = d_out.copy()
        d_input[self.last_input <= 0] = 0
        return d_input


class MaxPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None
        self.max_indices = None

    def forward(self, input):
        self.last_input = input
        channels, H, W = input.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        output = np.zeros((channels, H_out, W_out))
        # Stores (h, w) indices of max values for each window
        self.max_indices = np.zeros((channels, H_out, W_out, 2), dtype=np.int32)

        for c in range(channels):
            for i in range(H_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                for j in range(W_out):
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size

                    window = input[c, h_start:h_end, w_start:w_end]
                    max_val = np.max(window)
                    output[c, i, j] = max_val

                    # Find position of max value in window
                    h_max, w_max = np.unravel_index(np.argmax(window), window.shape)
                    self.max_indices[c, i, j] = [h_start + h_max, w_start + w_max]

        return output

    def backward(self, d_out):
        d_input = np.zeros_like(self.last_input)
        channels_out, H_out, W_out = d_out.shape

        for c in range(channels_out):
            for i in range(H_out):
                for j in range(W_out):
                    h_idx, w_idx = self.max_indices[c, i, j]
                    d_input[c, h_idx, w_idx] += d_out[c, i, j]

        return d_input


class AvgPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None  # Stores input for backward pass

    def forward(self, input):
        """Perform average pooling on the input tensor."""

        self.last_input = input
        channels, H, W = input.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        output = np.zeros((channels, H_out, W_out))

        for c in range(channels):
            for i in range(H_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                for j in range(W_out):
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size

                    window = input[c, h_start:h_end, w_start:w_end]
                    output[c, i, j] = np.mean(window)

        return output

    def backward(self, d_out):
        """Backpropagate gradients through the average pooling layer."""

        input = self.last_input
        channels, H, W = input.shape
        H_out, W_out = d_out.shape[1], d_out.shape[2]
        d_input = np.zeros_like(input)

        pool_area = self.pool_size * self.pool_size

        for c in range(channels):
            for i in range(H_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                for j in range(W_out):
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size

                    # Distribute gradient equally to all positions in the window
                    grad = d_out[c, i, j] / pool_area
                    d_input[c, h_start:h_end, w_start:w_end] += grad

        return d_input


class FCLayer:
    def __init__(self, input_size, output_size):
        """Fully Connected Layer implementation."""

        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * 0.1  # Xavier-like initialization
        self.biases = np.zeros((output_size, 1))
        self.last_input = None  # Cache for backpropagation

    def forward(self, input):
        """Forward pass through the layer."""

        self.last_input = input
        return np.dot(self.weights, input) + self.biases

    def backward(self, d_out, learning_rate):
        """Backward pass through the layer."""

        # Compute gradients
        d_weights = np.dot(d_out, self.last_input.T)
        d_biases = np.sum(d_out, axis=1, keepdims=True)  # Sum over batch

        # Compute gradient for previous layer
        d_input = np.dot(self.weights.T, d_out)

        # Update parameters
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input


def softmax(x):
    exp_shifted = np.exp(x - np.max(x))
    return exp_shifted / np.sum(exp_shifted, axis=0)


def cross_entropy_loss(probs, target_index):
    return -np.log(probs[target_index, 0] + 1e-9)  # 1e-9 для числовой стабильности


def softmax_cross_entropy_backward(probs, target_index):
    grad = probs.copy()
    grad[target_index] -= 1
    return grad


class Lenet5:
    def __init__(self):
        self.flat_shape = None
        self.conv1 = ConvLayer(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = ReLULayer()
        self.pool1 = AvgPoolLayer(pool_size=2, stride=2)
        self.conv2 = ConvLayer(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu2 = ReLULayer()
        self.pool2 = AvgPoolLayer(pool_size=2, stride=2)
        self.fc1 = FCLayer(input_size=16 * 5 * 5, output_size=120)
        self.relu3 = ReLULayer()
        self.fc2 = FCLayer(input_size=120, output_size=84)
        self.relu4 = ReLULayer()
        self.fc3 = FCLayer(input_size=84, output_size=10)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        self.flat_shape = out.shape
        out_flat = out.reshape(-1, 1)
        out = self.fc1.forward(out_flat)
        out = self.relu3.forward(out)
        out = self.fc2.forward(out)
        out = self.relu4.forward(out)
        out = self.fc3.forward(out)
        probs = softmax(out)
        return probs

    def backward(self, d_loss, learning_rate):
        d_out = self.fc3.backward(d_loss, learning_rate)
        d_out = self.relu4.backward(d_out)
        d_out = self.fc2.backward(d_out, learning_rate)
        d_out = self.relu3.backward(d_out)
        d_out = self.fc1.backward(d_out, learning_rate)
        d_out = d_out.reshape(self.flat_shape)
        d_out = self.pool2.backward(d_out)
        d_out = self.relu2.backward(d_out)
        d_out = self.conv2.backward(d_out, learning_rate)
        d_out = self.pool1.backward(d_out)
        d_out = self.relu1.backward(d_out)
        d_out = self.conv1.backward(d_out, learning_rate)
        return d_out


def softmax_crossentropy_with_logits(logits, labels):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    loss = -np.mean(np.sum(labels * np.log(probs + 1e-9), axis=1))
    return loss, probs


def grad_softmax_crossentropy(probs, labels):
    return (probs - labels) / labels.shape[0]

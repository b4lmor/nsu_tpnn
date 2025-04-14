import random
import time

import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, categorical=False, stop_criteria=95.0):
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()

            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            time2 = time.time()

            if test_data:
                accuracy = self.evaluate_regression(test_data, categorical)
                print(f"Epoch {j}: Accuracy = {accuracy:.2f}% | Time: {time2 - time1:.2f}s")
                if accuracy >= stop_criteria:
                    print(f"Stop criteria achieved!")
                    break
            else:
                print(f"Epoch {j} complete in {time2 - time1:.2f} seconds")

    def evaluate_regression(self, test_data, categorical):
        """Вычисляет точность в % для регрессии на основе относительной ошибки"""
        total_accuracy = 0
        n_samples = len(test_data)

        for x, y in test_data:
            output = self.feedforward(x.reshape(-1, 1))

            if categorical:
                # Для категориальных данных - сравниваем индексы максимумов
                predicted_class = np.argmax(output)
                true_class = np.argmax(y)
                total_accuracy += 100 if predicted_class == true_class else 0
            else:
                # Для регрессии - вычисляем относительную точность
                prediction = output[0][0]
                y_value = y[0] if isinstance(y, (np.ndarray, list)) else y

                if y_value != 0:
                    relative_error = abs(prediction - y_value) / abs(y_value)
                    accuracy = max(0, 1 - relative_error)
                    total_accuracy += accuracy * 100

        return total_accuracy / n_samples if n_samples > 0 else 0

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(-1, 1), y.reshape(-1, 1))
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
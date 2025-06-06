import copy

import numpy as np
from numpy.core.multiarray import array as array
from scipy.special import expit, softmax, log_softmax


class Module():
    def __init__(self):
        self.output = None
        self.training = True

    def compute_output(self, input, *args, **kwargs):
        raise NotImplementedError

    def compute_grad_input(self, input, *args, **kwargs):
        raise NotImplementedError

    def update_grad_parameters(self, input, *args, **kwargs):
        pass

    def __call__(self, input, *args, **kwargs):
        return self.forward(input, *args, **kwargs)

    def forward(self, input, *args, **kwargs):
        self.output = self.compute_output(input, *args, **kwargs)
        return self.output

    def backward(self, input, *args, **kwargs):
        grad_input = self.compute_grad_input(input, *args, **kwargs)
        self.update_grad_parameters(input, *args, **kwargs)
        return grad_input

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        pass

    def parameters(self):
        return []

    def parameters_grad(self):
        return []


class Criterion():
    def __init__(self):
        self.output = None

    def compute_output(self, input, target):
        raise NotImplementedError

    def compute_grad_input(self, input, target):
        raise NotImplementedError

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        self.output = self.compute_output(input, target)
        return self.output

    def backward(self, input, target):
        grad_input = self.compute_grad_input(input, target)
        return grad_input


class Optimizer():
    def __init__(self, module: Module):
        self.module = module
        self.state = {}

    def zero_grad(self):
        self.module.zero_grad()

    def step(self):
        raise NotImplementedError


class ReLU(Module):
    def compute_output(self, input):
        return np.where(input > 0, input, 0)

    def compute_grad_input(self, input, grad_output):
        return grad_output * np.where(input > 0, 1, 0)


class Tanh(Module):
    def compute_output(self, input):
        return np.tanh(input)

    def compute_grad_input(self, input, grad_output):
        return grad_output * (1 - (self.compute_output(input) ** 2))


class Sigmoid(Module):
    def compute_output(self, input):
        return expit(input)

    def compute_grad_input(self, input, grad_output):
        return grad_output * self.compute_output(input) * (1 - self.compute_output(input))


class LogSoftmax(Module):
    def compute_output(self, input):
        return log_softmax(input, axis=1)

    def compute_grad_input(self, input, grad_output):
        return grad_output - (np.sum(grad_output, axis=1, keepdims=True) * softmax(input, axis=1))


class CrossEntropyLoss(Criterion):
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input, target):
        return (-1 / input.shape[0]) * np.sum(
            input[np.arange(input.shape[0]), target] - np.log(np.sum(np.exp(input), axis=1)))

    def compute_grad_input(self, input, target):
        return (-1 / input.shape[0]) * (
                np.where(np.arange(input.shape[1]) == target[:, None], 1, 0) - softmax(input, axis=1))


class Linear(Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input):
        if self.bias is not None:
            return input @ self.weight.T + self.bias
        return input @ self.weight.T

    def compute_grad_input(self, input, grad_output):
        return grad_output @ self.weight

    def update_grad_parameters(self, input, grad_output):
        if self.bias is not None:
            self.grad_bias += np.sum(grad_output, axis=0)
        self.grad_weight += grad_output.T @ input

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def parameters_grad(self):
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]
        return [self.grad_weight]


class SGD(Optimizer):
    def __init__(self, module: Module, lr: float = 1e-3, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
        for param, grad, m in zip(parameters, gradients, self.state['m']):
            g = grad + self.weight_decay * param
            np.add(self.momentum * m, g, out=m)
            np.add(param, -self.lr * m, out=param)


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias = True, nonlinearity = 'relu'):
        super().__init__()
        self.weight_ih = np.random.uniform(-1, 1, (hidden_size, input_size)) / np.sqrt(hidden_size)
        self.weight_hh = np.random.uniform(-1, 1, (hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.bias_ih = np.random.uniform(-1, 1, hidden_size) / np.sqrt(hidden_size) if bias else None
        self.bias_hh = np.random.uniform(-1, 1, hidden_size) / np.sqrt(hidden_size) if bias else None

        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)

        self.cache = None

    def compute_output(self, input, hx):
        relu = ReLU()
        if self.bias_hh is not None:
            self.cache = input @ self.weight_ih.T + self.bias_ih + hx @ self.weight_hh.T + self.bias_hh
            return relu(self.cache)
        self.cache = input @ self.weight_ih.T + hx @ self.weight_hh.T
        return relu(self.cache)

    def compute_grad_input(self, input, hx, grad_output):
        relu = ReLU()
        ReLU_grad = relu.backward(self.cache, grad_output)
        return ReLU_grad @ self.weight_hh

    def update_grad_parameters(self, input, hx, grad_hx):
        relu = ReLU()
        ReLU_grad = relu.backward(self.cache, grad_hx)
        if self.bias_ih is not None:
            self.grad_bias_ih += np.sum(ReLU_grad, axis=0)
            self.grad_bias_hh += np.sum(ReLU_grad, axis=0)
        self.grad_weight_ih += ReLU_grad.T @ input
        self.grad_weight_hh += ReLU_grad.T @ hx

    def zero_grad(self):
        self.grad_weight_ih.fill(0)
        self.grad_weight_hh.fill(0)
        if self.bias_ih is not None:
            self.grad_bias_hh.fill(0)
            self.grad_bias_ih.fill(0)

    def parameters(self):
        if self.bias_hh is not None:
            return [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]
        return [self.weight_ih, self.weight_hh]

    def parameters_grad(self):
        if self.bias_hh is not None:
            return [self.grad_weight_ih, self.grad_weight_hh, self.grad_bias_ih, self.grad_bias_hh]
        return [self.grad_weight_ih, self.grad_weight_hh]


class RNN(Module):
    def __init__(self, input_size, hidden_size, bias = True, nonlinearity = 'relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.module = RNNCell(self.input_size, self.hidden_size, bias=self.bias, nonlinearity=nonlinearity)
        self.modules = None

        self.weight_ih_l0 = self.module.weight_ih
        self.weight_hh_l0 = self.module.weight_hh
        self.bias_ih_l0 = self.module.bias_ih if bias else None
        self.bias_hh_l0 = self.module.bias_hh if bias else None

        self.grad_weight_ih_l0 = np.zeros_like(self.weight_ih_l0)
        self.grad_weight_hh_l0 = np.zeros_like(self.weight_hh_l0)
        self.grad_bias_ih_l0 = np.zeros_like(self.bias_ih_l0)
        self.grad_bias_hh_l0 = np.zeros_like(self.bias_hh_l0)

    def compute_output(self, input, hx):
        self.modules = [copy.deepcopy(self.module) for _ in range(input.shape[1])]
        y = hx[0]
        for i in range(len(self.modules)):
            y = self.modules[i](input[:, i, :], y)
        return y

    def compute_grad_input(self, input, hx, grad_output):
        grad_input = grad_output[0]
        for i in range(len(self.modules) - 1, 0, -1):
            grad_input = self.modules[i].backward(input[:, i, :], self.modules[i - 1].output, grad_input)
        return self.modules[0].backward(input[:, 0, :], hx[0], grad_input)

    def update_grad_parameters(self, input, hx, grad_output):
        for module in self.modules:
            for rnn_grad, rnn_cell_grad in zip(self.parameters_grad(), module.parameters_grad()):
                rnn_grad += rnn_cell_grad

    def train(self):
        if self.modules is None:
            self.module.train()
            return
        for module in self.modules:
            module.train()

    def eval(self):
        if self.modules is None:
            self.module.eval()
            return
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        if self.modules is None:
            self.module.zero_grad()
            return
        self.grad_weight_ih_l0.fill(0)
        self.grad_weight_hh_l0.fill(0)
        if self.bias_ih_l0 is not None:
            self.grad_bias_hh_l0.fill(0)
            self.grad_bias_ih_l0.fill(0)

    def parameters(self):
        if self.bias_hh_l0 is not None:
            return [self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0]
        return [self.weight_ih_l0, self.weight_hh_l0]

    def parameters_grad(self):
        if self.bias_hh_l0 is not None:
            return [self.grad_weight_ih_l0, self.grad_weight_hh_l0, self.grad_bias_ih_l0, self.grad_bias_hh_l0]
        return [self.grad_weight_ih_l0, self.grad_weight_hh_l0]


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias = True):
        super().__init__()
        self.weight_ih = np.random.uniform(-1, 1, (4 * hidden_size, input_size)) / np.sqrt(hidden_size)
        self.weight_hh = np.random.uniform(-1, 1, (4 * hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.bias_ih = np.random.uniform(-1, 1, 4 * hidden_size) / np.sqrt(hidden_size) if bias else None
        self.bias_hh = np.random.uniform(-1, 1, 4 * hidden_size) / np.sqrt(hidden_size) if bias else None
        self.hidden_size = hidden_size

        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)

        self.ifgo_grad = None
        self.input_gate = None
        self.forget_gate = None
        self.output_gate = None
        self.gain_gate = None
        self.next_cell_state = None
        self.next_hidden_state = None

    def compute_output(self, input, hx, cx):
        tanh = Tanh()
        sigmoid = Sigmoid()
        A = input @ self.weight_ih.T + hx @ self.weight_hh.T
        if self.bias_hh is not None:
            A += self.bias_ih + self.bias_hh
        self.input_gate = sigmoid(A[:, 0: self.hidden_size])
        self.forget_gate = sigmoid(A[:, self.hidden_size: 2 * self.hidden_size])
        self.gain_gate = tanh(A[:, 2 * self.hidden_size: 3 * self.hidden_size])
        self.output_gate = sigmoid(A[:, 3 * self.hidden_size: 4 * self.hidden_size])
        self.next_cell_state = self.forget_gate * cx + self.input_gate * self.gain_gate
        self.next_hidden_state = self.output_gate * tanh(self.next_cell_state)
        return self.next_hidden_state, self.next_cell_state

    def compute_grad_input(self, input, hx, cx, grad_hx,
                           grad_cx):
        tanh = Tanh()
        tanh_grad = tanh.compute_grad_input(self.next_cell_state, grad_hx * self.output_gate)

        grad_next_cell_state = (grad_cx + tanh_grad) * self.forget_gate

        grad_input_gate = (grad_cx + tanh_grad) * self.gain_gate
        grad_input_gate *= self.input_gate * (1 - self.input_gate)

        grad_forget_gate = (grad_cx + tanh_grad) * cx
        grad_forget_gate *= self.forget_gate * (1 - self.forget_gate)

        grad_output_gate = grad_hx * tanh(self.next_cell_state)
        grad_output_gate *= self.output_gate * (1 - self.output_gate)

        grad_gain_gate = (grad_cx + tanh_grad) * self.input_gate
        grad_gain_gate *= (1 - (self.gain_gate ** 2))

        self.ifgo_grad = np.concatenate([grad_input_gate, grad_forget_gate, grad_gain_gate, grad_output_gate], axis=1)
        grad_next_hidden_state = self.ifgo_grad @ self.weight_hh
        return grad_next_hidden_state, grad_next_cell_state

    def update_grad_parameters(self, input, hx, cx, grad_hx, grad_cx):
        self.grad_weight_ih += self.ifgo_grad.T @ input
        self.grad_weight_hh += self.ifgo_grad.T @ hx
        if self.bias_hh is not None:
            self.grad_bias_ih += np.sum(self.ifgo_grad, axis=0)
            self.grad_bias_hh += np.sum(self.ifgo_grad, axis=0)

    def zero_grad(self):
        self.grad_weight_ih.fill(0)
        self.grad_weight_hh.fill(0)
        if self.bias_ih is not None:
            self.grad_bias_hh.fill(0)
            self.grad_bias_ih.fill(0)

    def parameters(self):
        if self.bias_hh is not None:
            return [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]
        return [self.weight_ih, self.weight_hh]

    def parameters_grad(self):
        if self.bias_hh is not None:
            return [self.grad_weight_ih, self.grad_weight_hh, self.grad_bias_ih, self.grad_bias_hh]
        return [self.grad_weight_ih, self.grad_weight_hh]


class LSTM(Module):
    def __init__(self, input_size, hidden_size, bias = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.module = LSTMCell(self.input_size, self.hidden_size, bias=self.bias)
        self.modules = None

        self.weight_ih_l0 = self.module.weight_ih
        self.weight_hh_l0 = self.module.weight_hh
        self.bias_ih_l0 = self.module.bias_ih if bias else None
        self.bias_hh_l0 = self.module.bias_hh if bias else None

        self.grad_weight_ih_l0 = np.zeros_like(self.weight_ih_l0)
        self.grad_weight_hh_l0 = np.zeros_like(self.weight_hh_l0)
        self.grad_bias_ih_l0 = np.zeros_like(self.bias_ih_l0)
        self.grad_bias_hh_l0 = np.zeros_like(self.bias_hh_l0)

    def compute_output(self, input, hx, cx):
        self.modules = [copy.deepcopy(self.module) for _ in range(input.shape[1])]
        hx = hx[0]
        cx = cx[0]
        for i in range(len(self.modules)):
            hx, cx = self.modules[i](input[:, i, :], hx, cx)
        return hx, cx

    def compute_grad_input(self, input, hx, cx, grad_hx):
        grad_hx = grad_hx[0]
        grad_cx = np.zeros_like(grad_hx)
        for i in range(len(self.modules) - 1, 0, -1):
            h_t, c_t = self.modules[i - 1].output
            grad_hx, grad_cx = self.modules[i].backward(input[:, i, :], h_t, c_t, grad_hx, grad_cx)
        return self.modules[0].backward(input[:, 0, :], hx[0], cx[0], grad_hx, grad_cx)

    def update_grad_parameters(self, input, hx, cx, grad_hx):
        for module in self.modules:
            for rnn_grad, rnn_cell_grad in zip(self.parameters_grad(), module.parameters_grad()):
                rnn_grad += rnn_cell_grad

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        if self.modules is None:
            self.module.train()
            return
        for module in self.modules:
            module.train()

    def eval(self):
        if self.modules is None:
            self.module.eval()
            return
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        if self.modules is None:
            self.module.zero_grad()
            return
        self.grad_weight_ih_l0.fill(0)
        self.grad_weight_hh_l0.fill(0)
        if self.bias_ih_l0 is not None:
            self.grad_bias_hh_l0.fill(0)
            self.grad_bias_ih_l0.fill(0)

    def parameters(self):
        if self.bias_hh_l0 is not None:
            return [self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0]
        return [self.weight_ih_l0, self.weight_hh_l0]

    def parameters_grad(self):
        if self.bias_hh_l0 is not None:
            return [self.grad_weight_ih_l0, self.grad_weight_hh_l0, self.grad_bias_ih_l0, self.grad_bias_hh_l0]
        return [self.grad_weight_ih_l0, self.grad_weight_hh_l0]


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, bias = True):
        super().__init__()
        self.weight_ih = np.random.uniform(-1, 1, (3 * hidden_size, input_size)) / np.sqrt(hidden_size)
        self.weight_hh = np.random.uniform(-1, 1, (3 * hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.bias_ih = np.random.uniform(-1, 1, 3 * hidden_size) / np.sqrt(hidden_size) if bias else None
        self.bias_hh = np.random.uniform(-1, 1, 3 * hidden_size) / np.sqrt(hidden_size) if bias else None
        self.hidden_size = hidden_size
        self.r = None
        self.z = None
        self.n = None
        self.A = None
        self.hidden_vec = None
        self.input_vec = None

        self.grad_A = None
        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)

        self.cache = None

    def compute_output(self, input, hx):
        sigmoid = Sigmoid()
        tanh = Tanh()
        self.input_vec = input @ self.weight_ih.T
        self.hidden_vec = hx @ self.weight_hh.T
        if self.bias_hh is not None:
            self.input_vec += self.bias_ih
            self.hidden_vec += self.bias_hh
        self.A = self.input_vec + self.hidden_vec
        self.r = sigmoid(self.A[:, 0: self.hidden_size])
        self.z = sigmoid(self.A[:, self.hidden_size: 2 * self.hidden_size])
        self.n = tanh(self.input_vec[:, 2 * self.hidden_size: 3 * self.hidden_size] +
                      self.r * self.hidden_vec[:, 2 * self.hidden_size: 3 * self.hidden_size])
        return (1 - self.z) * self.n + self.z * hx

    def compute_grad_input(self, input, hx, grad_output):
        grad_n = (grad_output * (1 - self.z) * (1 - (self.n ** 2)))
        grad_z = (grad_output * (-self.n + hx) * self.z * (1 - self.z))
        grad_r = ((grad_output * (1 - self.z) * (1 - (self.n ** 2))) *
                  self.hidden_vec[:, 2 * self.hidden_size: 3 * self.hidden_size] * self.r * (1 - self.r))
        self.grad_A = np.concatenate([grad_r, grad_z, grad_n], axis=1)
        grad_A_h = self.grad_A.copy()
        grad_A_h[:, 2 * self.hidden_size: 3 * self.hidden_size] *= self.r
        return grad_A_h @ self.weight_hh + grad_output * self.z

    def update_grad_parameters(self, input, hx, grad_hx):
        self.grad_weight_ih += self.grad_A.T @ input

        grad_A_weight_hh = self.grad_A.copy()
        grad_A_weight_hh[:, 2 * self.hidden_size: 3 * self.hidden_size] *= self.r
        self.grad_weight_hh += grad_A_weight_hh.T @ hx
        if self.bias_hh is not None:
            self.grad_bias_ih += np.sum(self.grad_A, axis=0)
            self.grad_bias_hh += np.sum(grad_A_weight_hh, axis=0)

    def zero_grad(self):
        self.grad_weight_ih.fill(0)
        self.grad_weight_hh.fill(0)
        if self.bias_ih is not None:
            self.grad_bias_hh.fill(0)
            self.grad_bias_ih.fill(0)

    def parameters(self):
        if self.bias_hh is not None:
            return [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]
        return [self.weight_ih, self.weight_hh]

    def parameters_grad(self):
        if self.bias_hh is not None:
            return [self.grad_weight_ih, self.grad_weight_hh, self.grad_bias_ih, self.grad_bias_hh]
        return [self.grad_weight_ih, self.grad_weight_hh]


class GRU(Module):
    def __init__(self, input_size, hidden_size, bias = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.module = GRUCell(self.input_size, self.hidden_size, bias=self.bias)
        self.modules = None

        self.weight_ih_l0 = self.module.weight_ih
        self.weight_hh_l0 = self.module.weight_hh
        self.bias_ih_l0 = self.module.bias_ih if bias else None
        self.bias_hh_l0 = self.module.bias_hh if bias else None

        self.grad_weight_ih_l0 = np.zeros_like(self.weight_ih_l0)
        self.grad_weight_hh_l0 = np.zeros_like(self.weight_hh_l0)
        self.grad_bias_ih_l0 = np.zeros_like(self.bias_ih_l0)
        self.grad_bias_hh_l0 = np.zeros_like(self.bias_hh_l0)

    def compute_output(self, input, hx):
        self.modules = [copy.deepcopy(self.module) for _ in range(input.shape[1])]
        y = hx[0]
        for i in range(len(self.modules)):
            y = self.modules[i](input[:, i, :], y)
        return y

    def compute_grad_input(self, input, hx, grad_output):
        grad_input = grad_output[0]
        for i in range(len(self.modules) - 1, 0, -1):
            grad_input = self.modules[i].backward(input[:, i, :], self.modules[i - 1].output, grad_input)
        return self.modules[0].backward(input[:, 0, :], hx[0], grad_input)

    def update_grad_parameters(self, input, hx, grad_output):
        for module in self.modules:
            for rnn_grad, rnn_cell_grad in zip(self.parameters_grad(), module.parameters_grad()):
                rnn_grad += rnn_cell_grad

    def train(self):
        if self.modules is None:
            self.module.train()
            return
        for module in self.modules:
            module.train()

    def eval(self):
        if self.modules is None:
            self.module.eval()
            return
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        if self.modules is None:
            self.module.zero_grad()
            return
        self.grad_weight_ih_l0.fill(0)
        self.grad_weight_hh_l0.fill(0)
        if self.bias_ih_l0 is not None:
            self.grad_bias_hh_l0.fill(0)
            self.grad_bias_ih_l0.fill(0)

    def parameters(self):
        if self.bias_hh_l0 is not None:
            return [self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0]
        return [self.weight_ih_l0, self.weight_hh_l0]

    def parameters_grad(self):
        if self.bias_hh_l0 is not None:
            return [self.grad_weight_ih_l0, self.grad_weight_hh_l0, self.grad_bias_ih_l0, self.grad_bias_hh_l0]
        return [self.grad_weight_ih_l0, self.grad_weight_hh_l0]


class RNN_Classifier(Module):
    def __init__(self, in_features, num_classes, hidden_size, module):
        super().__init__()
        self.input_size = in_features
        self.hidden_size = hidden_size
        self.h1 = None
        self.c1 = None
        self.encoder = module(in_features, hidden_size)
        self.head = Linear(hidden_size, num_classes)

    def compute_output(self, input):
        self.h1 = np.zeros((1, input.shape[0], self.hidden_size))
        if isinstance(self.encoder, LSTM):
            self.c1 = np.zeros((1, input.shape[0], self.hidden_size))
            out, _ = self.encoder(input, self.h1, self.c1)
        else:
            out = self.encoder(input, self.h1)
        return self.head(out)

    def compute_grad_input(self, input, grad_output):
        if isinstance(self.encoder, LSTM):
            return self.encoder.backward(input, self.h1, self.c1,
                                         self.head.backward(self.encoder.output[0], grad_output)[np.newaxis, :])[0]
        return self.encoder.backward(input, self.h1,
                                     self.head.backward(self.encoder.output, grad_output)[np.newaxis, :])

    def train(self):
        self.head.train()
        self.encoder.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()

    def zero_grad(self):
        self.encoder.zero_grad()
        self.head.zero_grad()

    def parameters(self):
        return self.encoder.parameters() + self.head.parameters()

    def parameters_grad(self):
        return self.encoder.parameters_grad() + self.head.parameters_grad()

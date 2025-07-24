import numpy as np

from micrograd.engine import Value

rng = np.random.default_rng()


class BaseModule:
    """Base class for all modules in the neural network."""

    def zero_grad(self) -> None:
        """Reset the gradients of the module's parameters."""

        for param in self.parameters():
            param.grad = 0.0

    def parameters(self) -> list[Value]:
        """Return the parameters of the module."""

        raise NotImplementedError('Subclasses should implement this method.')


class Neuron(BaseModule):
    """A simple neuron class to demonstrate the functionality of the Value class."""

    def __init__(self, num_inputs: int, nonlin = True) -> None:
        """Initialize the neuron with weights and bias."""

        self.w = [Value(rng.uniform(-1, 1), label=f'w{i}') for i in range(num_inputs)]
        self.b = Value(rng.uniform(-1, 1), label='b')
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        """Forward pass through the neuron."""

        activations: list[Value] = [
            x_i * w_i for x_i, w_i in zip(x, self.w, strict=True)
        ]
        y: Value = sum(activations, start=self.b)  # sum the weighted inputs and bias
        y.label = 'y'
        y = y.relu() if self.nonlin else y  # apply ReLU if nonlin is True

        return y

    def parameters(self) -> list[Value]:
        """Return the parameters of the neuron (weights and bias)."""

        return [*self.w, self.b]

    def __repr__(self) -> str:
        """Return a string representation of the Neuron."""

        return (
            f'{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})'
        )


class Layer(BaseModule):
    """A simple layer class to demonstrate the functionality of the Neuron class."""

    def __init__(self, num_inputs: int, num_neurons: int, **kwargs: bool) -> None:
        """Initialize the layer with a list of neurons."""

        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_neurons)]

    def __call__(self, x: list[Value]) -> list[Value]:
        """Forward pass through the layer."""

        y: list[Value] = [neuron(x) for neuron in self.neurons]

        return (
            y if len(y) > 1 else y[0]
        )  # return a single Value if there's only one neuron

    def parameters(self) -> list[Value]:
        """Return the parameters of the layer (weights and biases of all neurons)."""

        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self) -> str:
        """Return a string representation of the Layer."""

        return (
            f'Layer[{', '.join(repr(neuron) for neuron in self.neurons)}]'
        )


class MLP(BaseModule):
    """A simple multi-layer perceptron class to demonstrate the functionality of the Layer class."""

    def __init__(self, num_inputs: int, num_outputs: list[int]) -> None:
        """Initialize the MLP with an input layer, hidden layer, and output layer."""

        sz = [
            num_inputs,
            *num_outputs,
        ]  # size of each layer, first is input size, last is output size
        self.layers = [Layer(sz[i], sz[i + 1], nonlin = (i != len(num_outputs) - 1)) for i in range(len(num_outputs))]

    def __call__(self, x: list[Value]) -> list[Value]:
        """Forward pass through the MLP."""

        y = x
        for layer in self.layers:
            y = layer(y)

        return y

    def parameters(self) -> list[Value]:
        """Return the parameters of the MLP (weights and biases of all layers)."""

        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self) -> str:
        """Return a string representation of the MLP."""

        return (
            f'MLP[{', '.join(repr(layer) for layer in self.layers)}]'
        )

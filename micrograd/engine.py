"""A simple implementation of a computational graph for automatic differentiation."""

import math

from graphviz import Digraph


class Value:
    """A simple class to represent a value in a computational graph for backpropagation."""

    def __init__(
        self,
        data: float,
        _children: tuple['Value', ...] = (),
        _op: str = '',
        label: str = '',
    ) -> None:
        """Initialize the Value with a numeric value."""

        self.data: float = data
        self._prev: set[Value] = set(_children)
        self.grad: float = 0.0
        self._op: str = _op
        self.label: str = label

        # magic trick
        self._backward = lambda: None

    def __repr__(self) -> str:
        """Return a string representation of the Value."""

        return (
            f"Value(value={self.data:.5f}, grad={self.grad:.5f}, label='{self.label}')"
        )

    def __pow__(self, other: 'Value | float') -> 'Value':
        """Define exponentiation for Value objects."""

        # supports both Value and numeric exponentiation, addition to Karpathy's micrograd

        if not isinstance(other, Value):
            other = Value(other)

        res = Value(
            self.data**other.data,
            _children=(self, other),
            _op=f'{self.label}^{other.label}',
        )

        def _backward() -> None:
            # d/dx (x^y) = y * x^(y-1)
            # d/dy (x^y) = log(x) * x^y
            self.grad += other.data * (self.data ** (other.data - 1)) * res.grad
            if self.data > 0:
                other.grad += math.log(self.data) * res.data * res.grad
            else:
                other.grad += float('nan')  # log(x) is undefined for x <= 0

        res._backward = _backward

        return res

    def __rpow__(self, base: 'Value | float') -> 'Value':
        """Define reversed exponentiation for Value objects."""

        if not isinstance(base, Value):
            base = Value(base)

        return base ** self

    def __neg__(self) -> 'Value':
        """Define negation for Value objects."""

        return self * (-1)

    def __add__(self, other: 'Value | float') -> 'Value':
        """Define addition for Value objects."""

        if not isinstance(other, Value):
            other = Value(other)

        res = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward() -> None:
            # d/dx (x + y) = 1
            # d/dy (x + y) = 1
            self.grad += 1.0 * res.grad
            other.grad += 1.0 * res.grad

        res._backward = _backward

        return res

    def __radd__(self, other: 'Value | float') -> 'Value':
        """Define right addition for Value objects."""

        # other + self
        return self + other

    def __sub__(self, other: 'Value | float') -> 'Value':
        """Define subtraction for Value objects."""

        return self + (-other)

    def __rsub__(self, other: 'Value | float') -> 'Value':
        """Define right subtraction for Value objects."""

        # other - self or -self + other
        return -self + other

    def __mul__(self, other: 'Value | float') -> 'Value':
        """Define multiplication for Value objects."""

        if not isinstance(other, Value):
            other = Value(other)

        res = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward() -> None:
            # d/dx (x * y) = y + x * d/dx(y)
            # d/dy (x * y) = x + y * d/dy(x)
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad

        res._backward = _backward

        return res

    def __rmul__(self, other: 'Value | float') -> 'Value':
        """Define right multiplication for Value objects."""

        # other * self
        return self * other

    def __truediv__(self, other: 'Value | float') -> 'Value':
        """Define division for Value objects."""

        return self * (other ** (-1))

    def __rtruediv__(self, other: 'Value | float') -> 'Value':
        """Define right division for Value objects."""

        # other / self or other * (self ** -1)
        return other * (self**-1)

    def sin(self) -> 'Value':
        """Define the sine function for Value objects."""

        x: float = self.data
        s: float = math.sin(x)
        res = Value(s, _children=(self,), _op='sin', label='sin')

        def _backward() -> None:
            self.grad += math.cos(x) * res.grad

        res._backward = _backward

        return res

    def cos(self) -> 'Value':
        """Define the cosine function for Value objects."""

        x: float = self.data
        c: float = math.cos(x)
        res = Value(c, _children=(self,), _op='cos', label='cos')

        def _backward() -> None:
            self.grad += -math.sin(x) * res.grad

        res._backward = _backward

        return res

    def tan(self) -> 'Value':
        """Define the tangent function for Value objects."""

        x: float = self.data
        t: float = math.tan(x)
        res = Value(t, _children=(self,), _op='tan', label='tan')

        def _backward() -> None:
            self.grad += (1.0 / (math.cos(x) ** 2)) * res.grad

        res._backward = _backward

        return res

    def cot(self) -> 'Value':
        """Define the cotangent function for Value objects."""

        x: float = self.data
        c: float = 1.0 / math.tan(x)
        res = Value(c, _children=(self,), _op='cot', label='cot')

        def _backward() -> None:
            self.grad += (-1.0 / (math.sin(x) ** 2)) * res.grad

        res._backward = _backward

        return res

    def asin(self) -> 'Value':
        """Define the arcsine function for Value objects."""

        x: float = self.data
        if abs(x) > 1:
            raise ValueError('Arcsine undefined for |x| > 1')

        a: float = math.asin(x)
        res = Value(a, _children=(self,), _op='asin', label='asin')

        def _backward() -> None:
            self.grad += (1.0 / math.sqrt(1 - (x ** 2))) * res.grad

        res._backward = _backward

        return res

    def acos(self) -> 'Value':
        """Define the arccosine function for Value objects."""

        x: float = self.data
        if abs(x) > 1:
            raise ValueError('Arccosine undefined for |x| > 1')

        a: float = math.acos(x)
        res = Value(a, _children=(self,), _op='acos', label='acos')

        def _backward() -> None:
            self.grad += (-1.0 / math.sqrt(1 - (x ** 2))) * res.grad

        res._backward = _backward

        return res

    def atan(self) -> 'Value':
        """Define the arctangent function for Value objects."""

        x: float = self.data
        a: float = math.atan(x)
        res = Value(a, _children=(self,), _op='atan', label='atan')

        def _backward() -> None:
            self.grad += (1.0 / (1 + x**2)) * res.grad

        res._backward = _backward

        return res

    def relu(self) -> 'Value':
        """Define the ReLU activation function for Value objects."""

        x: float = self.data
        r: float = max(0.0, x)
        res = Value(r, _children=(self,), _op='relu', label='ReLU')

        def _backward() -> None:
            self.grad += (1.0 if x > 0 else 0.0) * res.grad

        res._backward = _backward

        return res

    def leaky_relu(self, alpha: float = 0.01) -> 'Value':
        """Define the Leaky ReLU activation function for Value objects."""

        x: float = self.data
        r: float = x if x > 0 else alpha * x
        res = Value(r, _children=(self,), _op='leaky_relu', label='Leaky ReLU')

        def _backward() -> None:
            self.grad += (1.0 if x > 0 else alpha) * res.grad

        res._backward = _backward

        return res

    def log(self) -> 'Value':
        """Define the natural logarithm for Value objects."""

        x: float = self.data
        if x <= 0:
            raise ValueError('Logarithm undefined for non-positive values.')

        l: float = math.log(x)
        res = Value(l, _children=(self,), _op='log', label='log')

        def _backward() -> None:
            self.grad += (1.0 / x) * res.grad

        res._backward = _backward

        return res

    def exp(self) -> 'Value':
        """Define the exponential function for Value objects."""

        x: float = self.data
        e: float = math.exp(x)
        res = Value(e, _children=(self,), _op='exp', label='exp')

        def _backward() -> None:
            self.grad += e * res.grad

        res._backward = _backward

        return res

    def erf(self) -> 'Value':
        """Define the error function for Value objects."""

        x: float = self.data
        e: float = math.erf(x)

        res = Value(e, _children=(self,), _op='erf', label='erf')

        def _backward() -> None:
            # d/dx erf(x) = (2 / sqrt(pi)) ** exp(-x^2)
            self.grad += (2 / math.sqrt(math.pi)) * math.exp(-x ** 2) * res.grad

        res._backward = _backward

        return res

    def tanh(self) -> 'Value':
        """Define the hyperbolic tangent function for Value objects."""

        x: float = self.data
        t: float = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        res = Value(t, _children=(self,), _op='tanh', label='tanh')

        def _backward() -> None:
            self.grad += (1.0 - t**2) * res.grad

        res._backward = _backward

        return res

    def trace(self) -> tuple[set['Value'], set['Value']]:
        """Trace the computational graph starting from the root Value."""

        nodes: set[Value] = set()
        edges: set[tuple[Value, Value]] = set()

        def dfs(value: Value) -> None:
            """Depth-first search to traverse the graph."""

            if value not in nodes:
                nodes.add(value)

                for child in value._prev:
                    edges.add((child, value))
                    dfs(child)

        dfs(self)

        return nodes, edges

    def topo_sort(self) -> list['Value']:
        """Perform a topological sort of the computational graph."""

        sorted_nodes: list[Value] = []
        visited: set[Value] = set()

        def dfs(node: Value) -> None:
            """Visit each node in the graph."""

            if node not in visited:
                visited.add(node)

                # Visit all children first
                for child in node._prev:
                    dfs(child)

                sorted_nodes.append(node)

        dfs(self)

        return sorted_nodes

    def backward(self) -> None:
        """Perform backpropagation to compute gradients."""

        topo_order = self.topo_sort()

        self.grad = 1.0
        for node in reversed(topo_order):
            node._backward()

    def draw_graph(self) -> Digraph:
        """Visualize the computational graph using Graphviz."""

        """Draw the computational graph using the trace function."""

        nodes, edges = self.trace()

        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

        for node in nodes:
            uid: str = str(id(node))

            dot.node(
                name=uid,
                label=f'{node.label} | data: {node.data:.4f} | grad: {node.grad:.4f}',
                shape='record',
            )

            if len(node._op) > 0:
                dot.node(name=uid + node._op, label=node._op, shape='circle')
                dot.edge(uid + node._op, uid)

        for node1, node2 in edges:
            dot.edge(str(id(node1)), str(id(node2)) + node2._op)

        return dot
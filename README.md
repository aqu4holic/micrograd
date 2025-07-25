# micrograd

a reimplementation of Karpathy's micrograd in Python. currently only supports scalar values.

- added `__pow__` and `__rpow__` methods for exponentiation and reversed exponentiation with `Value` class.
- added some geometric functions, and `erf` function.

all functions have been tested with some values, tests are in the `test` folder.

> [!CAUTION]
> with negative values, the logarithm is undefined, so the gradient will be nan, and the output will be in the complex plane.
for example:
```python
from micrograd.engine import Value

a = Value(-5)
b = x ** 0.5

b.backward()

print(b)
print(a)
print(type(a.grad))
```

output:
```
Value(value=0.00000+2.23607j, grad=1.00000, label='')
Value(value=-5.00000, grad=0.00000-0.22361j, label='')
<class 'complex'>
```

the `demo.ipynb` notebook is for the usage demo, the `scratchpad.ipynb` notebook is my scratchpad for testing various features and ideas.

# installation

```bash
uv init
uv sync --all-extras
```
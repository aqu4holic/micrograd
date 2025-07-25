import torch
from micrograd.engine import Value

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_new_ops():
    x = Value(2.0)
    y = Value(3.0)
    z = Value(4.0)

    t = 1.77 ** x + y ** 2.25 + z ** 3.5 + x ** y + y ** z + z ** x + (x ** y + z).log()**(x/y)
    t.backward()

    xmg, ymg, zmg, tmg = x, y, z, t

    x = torch.tensor([2.0]).double()
    y = torch.tensor([3.0]).double()
    z = torch.tensor([4.0]).double()
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True

    t: torch.Tensor = 1.77 ** x + y ** 2.25 + z ** 3.5 + x ** y + y ** z + z ** x + torch.log(x ** y + z)**(x/y)
    t.backward()

    xpt, ypt, zpt, tpt = x, y, z, t

    tol = 1e-6

    # forward pass went well
    assert abs(tmg.data - tpt.data.item()) < tol
    # backward pass went well
    assert abs(xmg.grad - xpt.grad.item()) < tol
    assert abs(ymg.grad - ypt.grad.item()) < tol
    assert abs(zmg.grad - zpt.grad.item()) < tol

def test_geometric_and_erf():
    x = Value(5.0)
    y = (2 ** (x)).erf().atan()
    z = Value(3.0)
    t = y.cos() + z.sin().asin()
    t.backward()

    xmg, zmg, tmg = x, z, t

    x = torch.tensor([5.0]).double()
    x.requires_grad = True
    y = (2 ** (x)).erf().atan()
    z = torch.tensor([3.0]).double()
    z.requires_grad = True
    t = y.cos() + z.sin().asin()
    t.backward()

    xpt, ypt, zpt, tpt = x, y, z, t

    tol = 1e-6

    # forward pass went well
    assert abs(tmg.data - tpt.data.item()) < tol
    # backward pass went well
    assert abs(xmg.grad - xpt.grad.item()) < tol
    assert abs(zmg.grad - zpt.grad.item()) < tol

test_sanity_check()
test_more_ops()
test_new_ops()
test_geometric_and_erf()
print("All tests passed!")
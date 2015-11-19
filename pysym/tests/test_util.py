from .. import Symbol, symarray


def test_symarray():
    arr = symarray('x', 2)
    assert arr[0] is Symbol('x_0')
    assert (arr[1] == Symbol('x_1')) is True

from .. import Matrix, Symbol, Number


def test_Matrix():
    x = Symbol('x')
    m = Matrix(1, 3, [x, x+1, 2*x - 3])
    assert m.shape == (1, 3)


def test_jacobian():
    symbols = x, y = Symbol('x'), Symbol('y')
    exprs = [3*x, 2*x*y + 3*y]

    def check(mat):
        assert mat.shape == (2, 2)
        assert (mat[0, 0] == Number(3)).evalb() is True
        assert (mat[0, 1] == Number(0)).evalb() is True
        assert (mat[1, 0] == 2*y).evalb() is True
        assert (mat[1, 1] == 2*x + 3).evalb() is True
        assert (mat[1, 1] == 2*x + 2).evalb() is False

    check(Matrix(1, 2, exprs).jacobian(symbols))
    check(Matrix(2, 1, exprs).jacobian(symbols))

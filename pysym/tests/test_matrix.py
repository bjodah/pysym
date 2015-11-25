from .. import Matrix, Symbol, Number


def test_Matrix():
    x = Symbol('x')
    m = Matrix(1, 3, [x, x+1, 2*x - 3])
    assert m.shape == (1, 3)


def _get_mat(nrow=1, ncol=2):
    symbols = x, y = Symbol('x'), Symbol('y')
    exprs = [3*x, 2*x*y + 3*y]
    return Matrix(nrow, ncol, exprs), symbols


def test_jacobian():

    mat, symbols = _get_mat()
    x, y = symbols

    def check(mat):
        assert mat.shape == (2, 2)
        assert (mat[0, 0] == Number(3)) is True
        assert (mat[0, 1] == Number(0)) is True
        assert (mat[1, 0] == 2*y) is True
        assert (mat[1, 1] == 2*x + 3) is True
        assert (mat[1, 1] == 2*x + 2) is False

    check(_get_mat(1, 2)[0].jacobian(symbols))
    check(_get_mat(2, 1)[0].jacobian(symbols))


def test_jacobian_array():
    import numpy as np
    mat, symbols = _get_mat()
    x, y = symbols
    jac = mat.jacobian(symbols)
    num_mat = jac.subs({x: Number(5), y: Number(7)}).evalf()
    nparr = np.array(num_mat)
    assert nparr.shape == (2, 2)
    assert np.allclose(nparr, [[3, 0], [2*7, 2*5 + 3]])

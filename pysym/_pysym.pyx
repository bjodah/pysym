from cpython cimport bool

import functools
import itertools
import math
import operator
import warnings
import weakref


def collect(sorted_args, collect_to):
    nargs = len(sorted_args)
    if nargs <= 1:
        return sorted_args
    prev = sorted_args[0]
    count = 1
    new_args = []

    def add(arg, count):
        if count == 1:
            new_args.append(arg)
        elif count > 1:
            new_args.append(collect_to.create(
                (arg, Number(count))
            ))

    for idx, arg in enumerate(sorted_args[1:], 1):
        is_last = (idx == (nargs - 1))
        if arg is prev:
            count += 1
            if is_last:
                add(arg, count)
            else:
                continue
        else:
            add(prev, count)
            if is_last:
                add(arg, 1)
            count = 1
            prev = arg
    return tuple(new_args)


def merge(args, mrg_cls=None):
    if mrg_cls is None:
        return args
    new_args = []
    merged = False
    for arg in args:
        if isinstance(arg, mrg_cls):
            new_args.extend(arg.args)
            merged = True
        else:
            new_args.append(arg)

    if merged:
        return merge(new_args, mrg_cls)
    else:
        return new_args

def _evalf(args):
    return tuple(arg if isinstance(arg, Number._NUMBER_TYPES)
                 else arg.evalf() for arg in args)


def merge_drop_sort_collect(args, collect_to, drop=(), mrg_cls=None):
    merged = merge(args, mrg_cls)
    return collect(
        sorted(filter(
            lambda x: not x.found_in(drop),
            merge(args, mrg_cls)
        )), collect_to)


cdef class _Basic:
    cdef int _hash
    cdef readonly tuple _args

    property args:
        def __get__(self):
            return self._args[2:]

    def __cinit__(self, *args):
        self._args = args
        self._hash = -1

    def create(self, args):
        return self._args[0].create(args)  # extra magic allowed

    cdef int args_hash(self):
        cdef int val = 0
        for arg in self._args:
            val += hash(arg)
        return val

    def __hash__(self):
        if self._hash == -1:
            self._hash = self.args_hash()
        return self._hash

    cpdef bool is_zero(self):
        return False

    cpdef bool is_atomic(self):
        return self._args[1]

    def __repr__(self):
        return '%s(%s)' % (self._args[0], ', '.join(
            repr(arg) for arg in self.args))

    def evalf(self):
        if self.is_atomic():
            return float(self.args[0])

    def diff(self, wrt):
        if self.is_atomic():
            if self._args[0] is Symbol:
                if wrt == self:
                    return One
                else:
                    return Zero
            return Zero
        return self._args[0].create(self.args).diff(wrt)

    cpdef bool has(self, instance):
        if not isinstance(instance, _Basic):
            instance = instance._obj
        if self.is_atomic():
            if instance is self:
                return True
            else:
                return False

        for arg in self.args:
            if arg.has(instance):
                return True
        return False

    cpdef bool found_in(self, flat_iterable):
        for elem_key in flat_iterable:
            if self == elem_key:
                return True
        return False

    cpdef object _subs(self, symb, repl):
        if self.is_atomic():
            return repl if self == symb else self

        if self.has(symb):
            if symb is self:
                raise ValueError("Impossible, bug!")
            else:
                return self.create(tuple([
                    repl if arg == symb else arg._subs(symb, repl)
                    for arg in self.args
                ]))
        else:
            return self

    def subs(self, subs_dict):
        result = self
        for key, val in subs_dict.items():
            result = result._subs(key, val)
        return result

    def expand(self):
        if self.is_atomic():
            return self
        return self.create(tuple(arg.expand() for arg in self.args))

    def __add__(self, other):
        return Add.create((Number.make(self), Number.make(other)))

    def __radd__(self, other):
        return self+other

    def __mul__(self, other):
        other = Number.make(other)
        return Mul.create((
            Number.make(self),
            Number.make(other)
        ))

    def __rmul__(self, other):
        return self*other

    def __pow__(base, exponent, modulo):
        exponent = Number.make(exponent)
        return Pow.create((base, exponent))

    def __truediv__(num, denom):
        num = Number.make(num)
        denom = Number.make(denom)
        return Fraction.create((num, denom))

    def __rtruediv__(denom, num):
        denom = Number.make(denom)
        return Fraction.create((num, denom))

    def __div__(num, denom):
        warnings.warn('Deprecated: Use "from __future__ import division"')
        denom = Number.make(denom)
        return Fraction.create((num, denom))

    def __rdiv__(denom, num):
        warnings.warn('Deprecated: Use "from __future__ import division"')
        denom = Number.make(denom)
        return Fraction.create((num, denom))

    def __sub__(self, other):
        return Sub.create((Number.make(self), Number.make(other)))

    def __rsub__(self, other):
        return self - other

    def __neg__(self):
        return -One * self

    def __richcmp__(self, other_, int op):
        cmp1 = self._args
        if isinstance(other_, _Basic):
            cmp2 = other_._args
        else:
            try:
                cmp2 = other_._obj._args
            except:
                cmp2 = Number.make(other_)._obj._args

        if op == 0:
            return cmp1 < cmp2
        elif op == 1:
            return cmp1 < cmp2
        elif op == 2:
            return cmp1 == cmp2
        elif op == 3:
            return cmp1 != cmp2
        elif op == 4:
            return cmp1 >= cmp2
        elif op == 5:
            return cmp1 > cmp2


class Basic(object):

    def __new__(cls, *args, atomic=False):
        instance = object.__new__(cls)
        _obj = _Basic(cls, atomic, *args)
        instance._obj = _obj
        instance.args = _obj.args
        return instance

    def __hash__(self): return hash(self._obj)
    def is_zero(self): return self._obj.is_zero()
    def is_atomic(self): return self._obj.is_atomic()

    def has(self, instance): return self._obj.has(instance)
    def found_in(self, flat_iterable): return self._obj.found_in(flat_iterable)
    def _subs(self, symb, repl): return self._obj._subs(symb, repl)
    def subs(self, subs_dict): return self._obj.subs(subs_dict)
    def expand(self): return self._obj.expand()
    def diff(self, wrt): return self._obj.diff(wrt)
    def __add__(self, other): return self._obj.__add__(other)
    def __radd__(self, other): return self._obj.__radd__(other)
    def __mul__(self, other): return self._obj.__mul__(other)
    def __rmul__(self, other): return self._obj.__rmul__(other)
    def __pow__(base, exponent): return base._obj.__pow__(exponent)
    def __truediv__(num, denom): return num._obj.__truediv__(denom)
    def __rtruediv__(denom, num): return denom._obj.__rtruediv__(num)
    def __div__(num, denom): return num._obj.__div__(denom)
    def __rdiv__(denom, num): return denom._obj.__rdiv__(num)
    def __sub__(self, other): return self._obj.__sub__(other)
    def __rsub__(self, other): return self._obj.__rsub__(other)
    def __neg__(self): return self._obj.__neg__()
    def _print_ccode(self): return self._obj._print_ccode()
    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(
            repr(arg) for arg in self.args))

    def __lt__(self, other): return self._obj < other
    def __le__(self, other): return self._obj <= other
    def __eq__(self, other): return self._obj == other
    def __ne__(self, other): return self._obj != other
    def __gt__(self, other): return self._obj > other
    def __ge__(self, other): return self._obj >= other


    @classmethod
    def from_args(cls, args):
        return cls(*args)

    @classmethod
    def create(cls, args):
        return cls(*args)  # extra magic allowed


class Relational(Basic):
    _rel_op = None
    _rel_op_str = None

    def evalb(self):
        return self._rel_op(*self.args)

    def __str__(self):
        return self._rel_op_str % self.args


class Eq(Relational):
    _rel_op = operator.__eq__
    _rel_op_str = '(%s == %s)'


class Ne(Relational):
    _rel_op = operator.__ne__
    _rel_op_str = '(%s != %s)'


class Lt(Relational):
    _rel_op = operator.__lt__
    _rel_op_str = '(%s < %s)'


class Le(Relational):
    _rel_op = operator.__le__
    _rel_op_str = '(%s <= %s)'


class Gt(Relational):
    _rel_op = operator.__gt__
    _rel_op_str = '(%s > %s)'


class Ge(Relational):
    _rel_op = operator.__ge__
    _rel_op_str = '(%s >= %s)'


class Not(Relational):
    _rel_op = operator.__not__
    _rel_op_str = '(not %s)'


class Atomic(Basic):

    __all_instances = weakref.WeakValueDictionary()

    def __new__(cls, arg):
        instance = Atomic.__all_instances.get(arg, None)
        if instance is None:
            instance = Basic.__new__(cls, arg, atomic=True)
            Atomic.__all_instances[arg] = instance
        return instance

    def found_in(self, flat_iterable):
        for elem in flat_iterable:
            if elem is self:
                return True
        return False


class Number(Atomic):

    _NUMBER_TYPES = (int, float)

    def __hash__(self):
        return hash(self.args[0])

    def is_zero(self):
        return self.args[0] == 0

    @classmethod
    def make(cls, arg):
        if isinstance(arg, cls._NUMBER_TYPES):
            return cls(arg)
        if hasattr(arg, 'dtype'):  # NumPy object
            return cls(arg)
        return arg

    def evalf(self):
        arg = self.args[0]
        if isinstance(arg, self._NUMBER_TYPES):
            return arg
        else:
            return float(arg)

    def __neg__(self):
        return Number(-self.args[0])

    def __str__(self):
        return str(self.args[0])

    def _print_ccode(self):
        return str(float(self.args[0]))  # integer division


class Symbol(Atomic):

    def __str__(self):
        return str(self.args[0])


Zero = Number(0)
One = Number(1)
Two = Number(2)


class Operator(Basic):
    _operator = None
    _op_str = None
    _op_cstr = None  # C-code
    _commutative = True

    def evalf(self):
        return self._operator(*_evalf(self.args))

    def __str__(self):
        return self._op_str % self.args

    def _print_ccode(self):
        return (self._op_cstr or self._op_str) % tuple(
            arg._print_ccode() for arg in self.args)

    def sorted(self):
        if self._commutative:
            return self.create(sorted(self.args))
        else:
            return self


class Reduction(Operator):

    @classmethod
    def create(cls, args):
        if len(args) == 1:
            return args[0]
        else:
            return super(Reduction, cls).create(args)

    def evalf(self):
        return functools.reduce(self._operator, _evalf(self.args))

    def __str__(self):
        return '(' + self._op_str.join(map(str, self.args)) + ')'

    def _print_ccode(self):
        return '(' + self._op_str.join(
            arg._print_ccode() for arg in self.args) + ')'


class Add(Reduction):

    _operator = operator.add
    _op_str = ' + '

    @classmethod
    def create(cls, args):
        args = tuple(filter(lambda x: x is not Zero, args))
        if len(args) == 0:
            return Zero
        else:
            args = merge_drop_sort_collect(args, Mul, (Zero, Mul(Zero)), Add)
            if len(args) == 0:
                return Zero
            else:
                return super(Add, cls).create(args)

    def diff(self, wrt):
        return self.create(tuple(arg.diff(wrt) for arg in self.args))

    def evalf(self):
        if len(self.args) == 0:
            return Zero
        else:
            return super(Add, self).evalf()

    def insert_mult(self, factor):
        return self.create(tuple(Mul.create((arg, factor))
                                 for arg in self.args))


class Mul(Reduction):

    _operator = operator.mul
    _op_str = '*'

    @classmethod
    def create(cls, args):
        if len(args) == 0:
            return One
        else:
            if Zero.found_in(args):
                return Zero
            else:
                return super(Mul, cls).create(
                    merge_drop_sort_collect(args, Pow, (One,), Mul))

    def diff(self, wrt):
        return Add.create(tuple(
            Mul.create(tuple(
                arg.diff(wrt) if i == idx else arg
                for i, arg in enumerate(self.args)))
            for idx in range(len(self.args))))

    def expand(self):
        for idx, arg in enumerate(self.args):
            if isinstance(arg, Add):
                if idx == 0:  # use of `create` guarantees len(args) > 1
                    return arg.insert_mult(Mul.create(
                        self.args[idx+1:])).expand()
                if idx > 0:  # absorb into first Add
                    summation = arg.insert_mult(Mul.create(self.args[:idx]))
                    return Mul.create(
                        (summation,) + self.args[idx + 1:]
                    ).expand()
        return self


class Binary(Operator):
    pass
    # def __init__(self, a, b):
    #     super(Binary, self).__init__(a, b)


class Sub(Binary):
    _operator = operator.sub
    _commutative = False
    _op_str = '(%s - %s)'

    @classmethod
    def create(cls, args):
        a, b = args  # a - b
        if a.is_zero():
            return -b
        if b.is_zero():
            return a
        if a == b:
            return Zero
        return cls(*args)

    def diff(self, wrt):
        return Sub.create((self.args[0].diff(wrt), self.args[1].diff(wrt)))


class Fraction(Binary):
    _operator = operator.truediv
    _commutative = False
    _op_str = '(%s/%s)'

    @classmethod
    def create(cls, args):
        instance = cls(*args)
        print(args)
        print(instance.args)
        if instance.args[1].is_zero():
            raise ZeroDivisionError
        else:
            if instance.args[0].is_zero():
                return Zero
            else:
                return instance

    def evalf(self):
        return float(self.args[0].evalf()) / float(self.args[1].evalf())

    def diff(self, wrt):
        a, b = self.args  # a/b
        return self.create((
            Sub.create((
                a.diff(wrt)*b,
                Mul.create((a, b.diff(wrt)))
            )),
            Pow.create((b, Two))
        ))


class Pow(Binary):

    _operator = operator.pow
    _commutative = False
    _op_str = '(%s**%s)'  # factorial has higher precedence (hence parenthesis)
    _op_cstr = 'pow(%s, %s)'

    def evalf(self):
        return self.args[0].evalf() ** self.args[1].evalf()

    def diff(self, wrt):
        base, exponent = self.args
        exponent *= log(base)
        if exponent.has(wrt):
            return Mul.create((exp(exponent), exponent.diff(wrt)))
        else:
            return Zero

    @classmethod
    def create(cls, args):
        base, exponent = args
        if exponent.is_zero():
            return One
        if base.is_zero():
            return Zero
        return cls(*args)


class ITE(Basic):

    # def __init__(self, cond, if_true, if_false):
    #     self.args = (cond, if_true, if_false)

    def _eval(self):
        return self.args[1] if self.args[0].evalb() else self.args[2]

    def evalf(self):
        return self._eval().evalf()

    def __str__(self):
        return '({1} if {0} else {2})'.format(*self.args)

    def _print_ccode(self):
        return '((%s) ? %s : %s)' % (arg._print_ccode() for arg in self.args)


class Function(Basic):
    _function = None
    _func_str = None

    def evalf(self):
        return self._function(*_evalf(self.args))

    def __str__(self):
        return (self._func_str or str(self._function)) + '(' + ', '.join(
            map(str, self.args)) + ')'


class Function1(Function):

    @staticmethod
    def _deriv(arg):
        raise NotImplementedError

    def diff(self, wrt):
        return Mul.create((self._deriv(self.args[0]),
                           self.args[0].diff(wrt)))


class gamma(Function1):

    _function = math.gamma
    _func_str = 'gamma'


class abs(Function1):
    _function = abs
    _func_str = 'abs'

    @staticmethod
    def _deriv(arg):
        return One


class exp(Function1):
    _function = math.exp
    _func_str = 'exp'

    @staticmethod
    def _deriv(arg):
        return exp(arg)


class sqrt(Function1):
    _function = math.sqrt
    _func_str = 'sqrt'

    @staticmethod
    def _deriv(arg):
        return 1/(2*sqrt(arg))


class log(Function1):
    _function = math.log
    _func_str = 'log'

    @staticmethod
    def _deriv(arg):
        return Pow(arg, -One)


class sin(Function1):
    _function = math.sin
    _func_str = 'sin'

    @staticmethod
    def _deriv(arg):
        return cos(arg)


class cos(Function1):
    _function = math.cos
    _func_str = 'sin'

    @staticmethod
    def _deriv(arg):
        return -sin(arg)


class tan(Function1):
    _function = math.tan
    _func_str = 'tan'

    @staticmethod
    def _deriv(arg):
        return One + tan(arg)**Two


class asin(Function1):
    _function = math.asin
    _func_str = 'asin'

    @staticmethod
    def _deriv(arg):
        return One/(One - arg**Two)**(One/Two)


class acos(Function1):
    _function = math.acos
    _func_str = 'acos'

    @staticmethod
    def _deriv(arg):
        return -asin._deriv(arg)


class atan(Function1):
    _function = math.atan
    _func_str = 'atan'

    @staticmethod
    def _deriv(arg):
        return One/(One + arg**Two)


class Vector(Basic):

    def __len__(self):
        return len(self.args)

    def diff(self, wrt):
        return self.__class__(tuple(arg.diff(wrt) for arg in self.args))

    def __iter__(self):
        return iter(self.args)

    def __getitem__(self, key):
        return self.args[key]


class Matrix(Basic):

    def __init__(self, nrows, ncols, source):
        if callable(source):
            callback = source
        else:
            def callback(ri, ci):
                try:
                    return source[ri, ci]
                except TypeError:
                    return source[ri*ncols + ci]
        elements = tuple([
            callback(ri, ci) for ri, ci in itertools.product(
                range(nrows),
                range(ncols))
        ])
        print(nrows, ncols, len(elements), type(elements), elements)
        super(Matrix, self).__init__(nrows, ncols *elements)

    def _subs(self, symb, repl):
        return self.__class__(self.nrows, self.ncols,
                              self.flatten()._subs(symb, repl))

    def __iter__(self):
        if self.shape[0] == 1:
            for ci in range(self.ncols):
                yield self[0, ci]
        elif self.shape[1] == 1:
            for ri in range(self.nrows):
                yield self[ri, 0]
        else:
            for ri in range(self.nrows):
                yield self.__class__(1, self.ncols, lambda i, j: self[ri, j])

    @property
    def nrows(self):
        return self.args[0]

    @property
    def ncols(self):
        return self.args[1]

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    def flatten(self):
        return Vector(*tuple(self[ri, ci] for ri, ci in itertools.product(
            range(self.nrows), range(self.ncols))))

    def _get_element(self, idx):
        return self.args[2+idx]

    def __getitem__(self, key):
        ri, ci = key
        return self._get_element(self.ncols*ri + ci)

    def jacobian(self, iterable):
        try:
            shape = iterable.shape
            if len(shape) > 2:
                raise ValueError
        except AttributeError:
            iterable = tuple(iterable)
        else:
            if len(shape) == 2:
                if shape[0] != 1 and shape[1] != 1:
                    raise ValueError('need column or row vector')
                if shape[0] == 1:
                    iterable = tuple(iterable[0, i] for i in range(shape[1]))
                else:
                    iterable = tuple(iterable[i, 0] for i in range(shape[0]))
        if self.ncols != 1 and self.nrows != 1:
            raise TypeError('jacobian only defined for row or column matrices')
        if self.ncols == 1:
            return self.__class__(
                self.nrows, len(iterable),
                lambda ri, ci: self._get_element(ri).diff(iterable[ci]))
        elif self.nrows == 1:
            return self.__class__(
                max(self.shape), len(iterable),
                lambda ri, ci: self._get_element(ri).diff(iterable[ci]))

    def evalf(self):
        return [[self[ri, ci].evalf() for ci in range(self.ncols)]
                for ri in range(self.nrows)]


def _wrap_numbers(func):
    @functools.wraps(func)
    def f(*args):
        new_args = tuple(map(Number.make, args))
        return func(*new_args)
    return f

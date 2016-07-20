# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function)

import functools
import itertools
import math
import operator
import warnings
import weakref


def _wrap_numbers(func):
    @functools.wraps(func)
    def f(*args):
        new_args = tuple(map(Number.make, args))
        return func(*new_args)
    return f


class _deprecated(object):
    def __init__(self, msg):
        self.msg = msg

    def __call__(self, func):
        @functools.wraps(func)
        def f(*args, **kwargs):
            warnings.warn('Deprecation warning: ' + self.msg)
            return func(*args, **kwargs)
        return f


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


def merge_drop_sort_collect(args, collect_to, drop=(), mrg_cls=None):
    return collect(
        sorted(filter(
            lambda x: not x.found_in(drop),
            merge(sorted(args), mrg_cls)
        )), collect_to)


@functools.total_ordering
class Basic(object):

    __slots__ = ('args',)

    def is_atomic(self):
        return False

    def is_zero(self):
        return False

    def __init__(self, *args):
        self.args = args

    @classmethod
    def create(cls, args):
        return cls(*args)  # extra magic allowed

    def __hash__(self):
        return hash(self.args)

    def __lt__(self, other):
        other = Number.make(other)
        typ1, typ2 = type(self), type(other)
        if typ1 is typ2:
            if self.is_atomic():
                return self.args[0] < other.args[0]
            else:
                for a1, a2 in zip(self.args, other.args):
                    if a1 < a2:
                        return True
                return False
        else:
            return typ1.__name__ < typ2.__name__

    def __eq__(self, other):
        other = Number.make(other)
        typ1, typ2 = type(self), type(other)
        if typ1 is typ2:
            if self.is_atomic():
                return self.args[0] == other.args[0]
            else:
                for a1, a2 in zip(self.args, other.args):
                    if not a1 == a2:
                        return False
                return True
        else:
            return False

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(
            repr(arg) for arg in self.args))

    def _print_ccode(self):
        return str(self)

    def has(self, instance):
        for arg in self.args:
            if arg.has(instance):
                return True
        return False

    def found_in(self, flat_iterable):
        for elem_key in flat_iterable:
            if self == elem_key:
                return True
        return False

    def _subs(self, symb, repl):
        if self.has(symb):
            if symb is self:
                raise ValueError("Impossible, bug!")
            else:
                return self.create(tuple(
                    repl if arg is symb else arg._subs(symb, repl)
                    for arg in self.args
                ))
        else:
            return self

    def subs(self, subs_dict):
        result = self
        for key, val in subs_dict.items():
            result = result._subs(key, val)
        return result

    def expand(self):
        return self.create(tuple(arg.expand() for arg in self.args))

    @_wrap_numbers
    def __add__(self, other):
        return Add.create((self, other))

    def __radd__(self, other):
        return self+other

    @_wrap_numbers
    def __mul__(self, other):
        return Mul.create((self, other))

    def __rmul__(self, other):
        return self*other

    @_wrap_numbers
    def __pow__(base, exponent):
        return Pow.create((base, exponent))

    @_wrap_numbers
    def __truediv__(num, denom):
        return Fraction.create((num, denom))

    @_wrap_numbers
    def __rtruediv__(denom, num):
        return Fraction.create((num, denom))

    @_deprecated('Use "from __future__ import division"')
    @_wrap_numbers
    def __div__(num, denom):
        return Fraction.create((num, denom))

    @_deprecated('Use "from __future__ import division"')
    @_wrap_numbers
    def __rdiv__(denom, num):
        return Fraction.create((num, denom))

    @_wrap_numbers
    def __sub__(self, other):
        return Sub.create((self, other))

    @_wrap_numbers
    def __rsub__(self, other):
        return Sub.create((other, self))

    def __neg__(self):
        return -One * self


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
    __slots__ = ('args', '__all_Atomic_instances',)

    def is_atomic(self):
        return True

    def __new__(cls, arg):
        instance = Atomic.__all_instances.get(arg, None)
        if instance is None:
            instance = object.__new__(cls)
            instance.args = (arg,)
            Atomic.__all_instances[arg] = instance
        return instance

    def has(self, instance):
        if instance is self:
            return True

    def found_in(self, flat_iterable):
        for elem in flat_iterable:
            if elem is self:
                return True
        return False

    def _subs(self, symb, repl):
        return repl if self is symb else self

    def expand(self):
        return self


class Number(Atomic):

    _NUMBER_TYPES = (int, float)

    def __abs__(self):
        return -self if self < 0 else self

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

    def diff(self, wrt):
        return Zero

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

    def diff(self, instance):
        if instance is self:
            return One
        else:
            return Zero

    def __str__(self):
        return str(self.args[0])


def Dummy():
    Dummy.counter -= 1
    return Symbol('Dummy' + str(Dummy.counter - 1))

Dummy.counter = 1

Zero = Number(0)
One = Number(1)
Two = Number(2)
nan = Number(float('nan'))


class Operator(Basic):
    _operator = None
    _op_str = None
    _op_cstr = None  # C-code
    _commutative = True

    def evalf(self):
        return self._operator(*tuple(arg.evalf() for arg in self.args))

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
        return functools.reduce(self._operator, (
            arg.evalf() for arg in self.args))

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
            return super(Add, cls).create(
                merge_drop_sort_collect(args, Mul,
                                        (Zero, Mul(Zero)), Add))

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

    def __init__(self, a, b):
        super(Binary, self).__init__(a, b)


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
        if isinstance(a, Number) and isinstance(b, Number):
            return Number.make(a.args[0] - b.args[0])
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
        if instance.args[1].is_zero():
            raise ZeroDivisionError
        else:
            if instance.args[0].is_zero():
                return Zero
            else:
                return instance

    def evalf(self):
        return self.args[0].evalf() / self.args[1].evalf()

    def diff(self, wrt):
        a, b = self.args  # a/b
        return self.create((
            Sub.create((
                a.diff(wrt)*b,
                Mul.create((a, b.diff(wrt)))
            )),
            Pow.create((b, Two))
        ))
        # return (self.args[0] * self.args[1]**-One).diff(wrt)


class Pow(Binary):

    _operator = operator.pow
    _commutative = False
    _op_str = '(%s**%s)'  # factorial has higher precedence (hence parenthesis)
    _op_cstr = 'pow(%s, %s)'

    def evalf(self):
        return self.args[0].evalf() ** self.args[1].evalf()

    def diff(self, wrt):
        base, exponent = self.args
        in_base = base.has(wrt)
        in_exponent = exponent.has(wrt)
        if in_base:
            if in_exponent:
                pass
            else:
                return Mul.create((
                    exponent, Pow.create((
                        base, Sub.create((
                            exponent, One
                        ))
                    )),
                    base.diff(wrt)
                ))
        else:
            if in_exponent:
                pass
            else:
                return Zero

        return exp.create((
            Mul.create((
                log.create((
                    base,
                )),
                exponent
            )),
        )).diff(wrt)

        # exponent *= log(base)
        # if exponent.has(wrt):
        #     return Mul.create((exp(exponent), exponent.diff(wrt)))
        # else:
        #     return Zero

    @classmethod
    def create(cls, args):
        base, exponent = args
        if exponent.is_zero():
            return One
        elif exponent == One:
            return base
        if base.is_zero():
            return Zero
        return cls(*args)


class ITE(Basic):

    def __init__(self, cond, if_true, if_false):
        self.args = (cond, if_true, if_false)

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
        return self._function(*tuple(arg.evalf() for arg in self.args))

    def __str__(self):
        return (self._func_str or str(self._function)) + '(' + ', '.join(
            map(str, self.args)) + ')'


class Function1(Function):

    @_wrap_numbers
    def __init__(self, arg):
        self.args = (arg,)

    @staticmethod
    def _deriv(arg):
        raise NotImplementedError

    def diff(self, wrt):
        return Mul.create((self._deriv(self.args[0]),
                           self.args[0].diff(wrt)))


class gamma(Function1):

    _function = math.gamma
    _func_str = 'gamma'


class Abs(Function1):
    _function = abs
    _func_str = 'Abs'

    @staticmethod
    def _deriv(arg):
        return ITE(Lt(arg, Zero), -One,
                   ITE(Gt(arg, Zero), One, nan))


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

    def __init__(self, *args):
        self.args = args

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
                    return Number.make(source[ri, ci])
                except (TypeError, IndexError):
                    return Number.make(source[ri*ncols + ci])
        self.args = (nrows, ncols) + tuple(
            callback(ri, ci) for ri, ci in itertools.product(
                range(nrows), range(ncols))
        )

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

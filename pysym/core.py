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


@functools.total_ordering
class BasicComparator(object):
    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        typ1, typ2 = type(self.obj), type(other.obj)
        if typ1 is typ2:
            if self.obj.is_atomic:
                return self.obj.args[0] < other.obj.args[0]
            else:
                for a1, a2 in zip(self.obj.args, other.obj.args):
                    if BasicComparator(a1) < BasicComparator(a2):
                        return True
                return False
        else:
            return typ1.__name__ < typ2.__name__

    def __eq__(self, other):
        typ1, typ2 = type(self.obj), type(other.obj)
        if typ1 is typ2:
            if self.obj.is_atomic:
                return self.obj.args[0] == other.obj.args[0]
            else:
                for a1, a2 in zip(self.obj.args, other.obj.args):
                    if not BasicComparator(a1) == BasicComparator(a2):
                        return False
                return True
        else:
            return False


def _collect(args, collect_to, drop=()):
    count = 0
    previous = None
    new_args = []
    for arg in sorted(args, key=BasicComparator):
        if arg.found_in(drop):
            continue
        if count > 0:
            if arg is previous:
                count += 1
                continue
            else:
                if count > 1:
                    new_args.append(collect_to.create(
                        (previous, Number(count))))
                else:
                    new_args.append(previous)
        count = 1
        previous = arg
    if count == 1:
        new_args.append(previous)
    elif count > 1:
        new_args.append(collect_to(previous, Number(count)))
    return tuple(new_args)


def _associative(Cls, lhs, rhs, collect_to=None, drop=()):
    lhs_cls = isinstance(lhs, Cls)
    rhs_cls = isinstance(rhs, Cls)
    if not lhs_cls and not rhs_cls:
        args = lhs, rhs
    else:
        if lhs_cls and rhs_cls:
            args = lhs.args + rhs.args
        elif isinstance(lhs, Cls):
            args = lhs.args + (rhs,)
        elif isinstance(rhs, Cls):
            args = rhs.args + (lhs,)
        if collect_to is not None:
            args = _collect(args, collect_to, drop)

    return Cls.create(args)


class Basic(object):

    __slots__ = ('args',)

    @property
    def is_atomic(self):
        return False

    def is_zero(self):
        return False

    def __init__(self, *args):
        self.args = args

    @classmethod
    def from_args(cls, args):
        return cls(*args)

    @classmethod
    def create(cls, args):
        return cls(*args)  # extra magic allowed

    def __hash__(self):
        return hash(self.args)

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
        sort_key = BasicComparator(self)
        for elem_key in map(BasicComparator, flat_iterable):
            if sort_key == elem_key:
                return True
        return False

    def _subs(self, symb, repl):
        if symb is self:
            raise ValueError("Impossible, bug!")
        else:
            return self.__class__(*tuple(
                repl if arg is symb else arg._subs(symb, repl)
                for arg in self.args
            ))

    def subs(self, subs_dict):
        result = self
        for key, val in subs_dict.items():
            result = result._subs(key, val)
        return result

    @_wrap_numbers
    def __add__(self, other):
        return _associative(Add, self, other, Mul, (Zero, Mul(Zero)))

    def __radd__(self, other):
        return self+other

    @_wrap_numbers
    def __mul__(self, other):
        return _associative(Mul, self, other, Pow, (One,))

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
        neg = -One*other
        instance = self + neg
        return instance

    def __rsub__(self, other):
        return self - other

    @_wrap_numbers
    def __neg__(self):
        return -One * self

    @_wrap_numbers
    def __eq__(self, other):
        return Eq(self, other)

    @_wrap_numbers
    def __ne__(self, other):
        return Ne(self, other)

    @_wrap_numbers
    def __lt__(self, other):
        return Lt(self, other)

    @_wrap_numbers
    def __le__(self, other):
        return Le(self, other)

    @_wrap_numbers
    def __gt__(self, other):
        return Gt(self, other)

    @_wrap_numbers
    def __ge__(self, other):
        return Ge(self, other)


class Relational(Basic):
    _rel_op = None
    _rel_op_str = None

    def evalb(self):
        return self._rel_op(*map(BasicComparator, self.args))

    def __str__(self):
        return self._rel_op_str % self.args


class Eq(Relational):
    _rel_op = operator.__eq__
    _rel_op_str = '(%s != %s)'


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
    _rel_op_str = '(!%s)'


class Atomic(Basic):

    __all_instances = weakref.WeakValueDictionary()
    __slots__ = ('args', '__all_Atomic_instances',)

    @property
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
        return self


class Number(Atomic):

    _NUMBER_TYPES = (int, float)

    def is_zero(self):
        return self.args[0] == 0

    @classmethod
    def make(cls, arg):
        if isinstance(arg, cls._NUMBER_TYPES):
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

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.args[0] == other
        return super(Number, self).__eq__(other)

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


Zero = Number(0)
One = Number(1)
Two = Number(2)


class Operator(Basic):
    _operator = None
    _op_str = None
    _op_cstr = None  # C-code
    _commutative = True

    def evalf(self):
        return self._operator(arg.evalf() for arg in self.args)

    def __str__(self):
        return self._op_str % self.args

    def _print_ccode(self):
        return (self._op_cstr or self._op_str) % tuple(
            arg._print_ccode() for arg in self.args)

    def sorted(self):
        if self._commutative:
            return self.__class__(*sorted(self.args, key=BasicComparator))
        else:
            return self


class Unary(Operator):

    def __init__(self, a):
        super(Unary, self).__init__(a)


class Binary(Operator):

    def __init__(self, a, b):
        super(Binary, self).__init__(a, b)


class Reduction(Operator):

    @classmethod
    def create(cls, args):
        instance = cls(*args)
        if len(instance.args) == 1:
            return instance.args[0]
        else:
            return instance

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

    def __init__(self, *args):
        self.args = _collect(args, Mul, (Zero, Mul(Zero)))

    def __iadd__(self, other):
        self.args += (other,)

    def diff(self, wrt):
        return self.create(tuple(arg.diff(wrt) for arg in self.args))

    def evalf(self):
        if len(self.args) == 0:
            return Zero
        else:
            return super(Add, self).evalf()


class Mul(Reduction):

    _operator = operator.mul
    _op_str = '*'

    def __init__(self, *args):
        if Zero.found_in(args):
            self.args = (Zero,)
        else:
            self.args = _collect(args, Pow, (One,))

    def __imul__(self, other):
        self.args += (other,)

    def diff(self, wrt):
        return Add.create(tuple(
            Mul.create(tuple(
                arg.diff(wrt) if i == idx else arg
                for i, arg in enumerate(self.args)))
            for idx in range(len(self.args))))


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
        return (self.args[0] * self.args[1]**-One).diff(wrt)


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
            return exp(exponent)*exponent.diff(wrt)
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

    def __init__(self, cond, if_true, if_false):
        self.args = (cond, if_true, if_false)

    def evalb(self):
        return self.args[1] if self.args[0].evalb() else self.args[2]

    def evalf(self):
        return self.evalb().evalf()

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


class Matrix(Basic):

    def __init__(self, nrows, ncols, callback):
        self.args = (nrows, ncols) + tuple(
            callback(ri, ci) for ri, ci in itertools.product(
                range(nrows), range(ncols))
        )

    @property
    def nrows(self):
        return self.args[0]

    @property
    def ncols(self):
        return self.args[1]

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    def _get_element(self, idx):
        return self.args[2+idx]

    def __getitem__(self, key):
        ri, ci = key
        return self._get_element(self.ncols*ri + ci)

    def jacobian(self, iterable):
        iterable = tuple(iterable)
        if self.ncols != 1:
            raise NotImplementedError
        return self.__class__(
            self.nrows, len(iterable),
            lambda ri, ci: self._get_element[ri].diff(iterable[ci]))

    def evalf(self):
        return [[self[ri, ci].evalf() for ci in range(self.ncols)]
                for ri in range(self.nrows)]


def lambdify(args, exprs):
    def f(*inp):
        subsd = dict(zip(args, inp))
        return [expr.subs(subsd).evalf() for expr in exprs]

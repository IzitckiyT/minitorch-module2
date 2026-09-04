"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x, y):
    return x * y

def id(x):
    return x

def add(x, y):
    return x + y

def neg(x):
    return -1.0 * x

def lt(x, y):
    return 1.0 if x < y else 0.0

def eq(x, y):
    return 1.0 if x == y else 0.0

def max(x, y):
    return x if x > y else y

def is_close(x, y):
    return abs(x - y) < 1e-2

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

def relu(x):
    return max(0.0, x)

def log(x):
    return math.log(x)

def exp(x):
    return math.exp(x)

def inv(x):
    return 1.0 / x

def log_back(x, y):
    return y / x

def inv_back(x, y):
    return -y / (x * x)

def relu_back(x, y):
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(f, iterable):
    return [f(x) for x in iterable]

def zipWith(f, iterable1, iterable2):
    return [f(x, y) for x, y in zip(iterable1, iterable2)]

def reduce(f, iterable):
    first = False
    result = None
    for x in iterable:
        if not first:
            result = x
            first = True
        else:
            result = f(result, x)
    return result

def negList(iterable):
    return map(neg, iterable)

def addLists(iterable1, iterable2):
    return zipWith(add, iterable1, iterable2)

def sum(iterable):
    result = reduce(add, iterable)
    return result if result is not None else 0.0

def prod(iterable):
    result = reduce(mul, iterable)
    return result if result is not None else 1.0

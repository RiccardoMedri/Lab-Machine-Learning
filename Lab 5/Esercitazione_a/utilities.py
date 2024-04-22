import math

def power(x: float) -> float:
    return x ** 2

def sqrt(x: float) -> float:
    return math.sqrt(x)

def offset(x: float) -> float:
    return x+100

NUMBER_TRANSFORMATION = lambda x: offset(sqrt(power(x)))
from typing import Union, TypeVar, TypeAlias, List
from collections.abc import Mapping
from random import choice, choices

T = TypeVar('T')
Number: TypeAlias = Union[int, float]


def argmax(mapping: Mapping[T, Number]) -> T:
    M = max(mapping.values())
    return choice([k for k, v in mapping.items() if v == M])


def avg(it: List[Number]):
    return sum(it) / len(it)


def get_random_weighted_key(mapping: Mapping[T, Number]) -> T:
    return choices(list(mapping.keys()), list(mapping.values()))[0]

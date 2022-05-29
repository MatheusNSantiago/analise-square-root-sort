import numba as nb
import numpy as np
from math import ceil, floor, sqrt
from tqdm import trange


@nb.experimental.jitclass([("V", nb.int64[:]), ("size", nb.int64)])
class Parte:
    def __init__(self, V, size):
        self.V = V
        self.size = size

    def __setitem__(self, index, new_value):
        if index < 0:
            index = self.size + index

        self.V[index] = new_value

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError("Index fora da range")
        if index < 0:
            index = self.size + index

        return self.V[index]

    def pop(self):
        self.size -= 1
        return self.V[self.size]


@nb.njit
def particiona_array(array: np.ndarray) -> list[Parte]:
    n = array.size

    max_len_parte = floor(sqrt(n))
    n_partes = ceil(n / max_len_parte)
    resto = n % max_len_parte

    partes = []
    for i in range(n_partes):
        start = i * max_len_parte
        nxt = (i + 1) * max_len_parte

        if (i != n_partes - 1) or (resto == 0):
            p = Parte(array[start:nxt], max_len_parte)
        else:
            p = Parte(array[start:], resto)

        partes.append(p)

    return partes


def progress_bar(
    iterable,
    desc,
    bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{rate_inv_fmt}]  ",
):
    return trange(
        iterable,
        unit=" ordenação",
        desc=desc,
        bar_format=bar_format,
    )

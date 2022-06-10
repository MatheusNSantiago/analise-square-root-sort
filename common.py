import numba as nb
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt
import numpy as np

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

  
    len_parte = floor(sqrt(n))
    n_partes = ceil(n / len_parte)
    resto = n % len_parte

    partes = []
    for i in range(n_partes):
        start = i * len_parte
        nxt = (i + 1) * len_parte

        if (i != n_partes - 1) or (resto == 0):
            p = Parte(array[start:nxt], len_parte)
        else:
            p = Parte(array[start:], resto)

        partes.append(p)

    return partes

# |────────────────────────────────────| Versão simplificado para por no projeto |────────────────────────────────────|

# def particiona_array(V):
#     n = len(V)

#     len_parte = floor(sqrt(n))
#     n_partes = ceil(sqrt(n))

#     partes = []

#     for i in range(n_partes):
#         start = i * len_parte
#         nxt = (i + 1) * len_parte

#         if i != n_partes - 1:
#             p = V[start:nxt]
#         else:
#             p = V[start:]

#         partes.append(p)

#     return partes




def plot_array(resultados, sizes):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 9), tight_layout=True)
    res_quadratico = [round(np.mean(i), 3) for i in resultados["quadratico"].values()]
    res_heap = [round(np.mean(i), 3) for i in resultados["heap"].values()]

    ax1.plot(sizes, res_quadratico, label="Método Quadrático")

    ax1.plot(sizes, res_heap, label="Heap")

    ax1.legend()
    ax1.set_ylabel("Tempo (s)", labelpad=5)
    ax1.set_xlabel("Tamanho da entrada", labelpad=0)
    ax1.set_xscale("log")

    x = np.linspace(sizes[0], sizes[-1], 150)
    y = np.interp(x, sizes, res_quadratico)
    ax2.plot(x, y, label="Interpolação Método Quadrático ")

    x = np.linspace(sizes[0], sizes[-1], 150)
    y = np.interp(x, sizes, res_heap)

    ax2.plot(x, y, label="Heap Interpolação")
    ax2.legend()
    ax2.set_ylabel("Tempo (s)", labelpad=5)
    ax2.set_xlabel("Tamanho da entrada", labelpad=0)
    ax2.set_xscale("log")

    fig.subplots_adjust(bottom=0.85)


    plt.show()
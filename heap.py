from collections import namedtuple
from typing import List, Any
from common import particiona_array
from numba import njit, jit, int64
import numba as nb
import numpy as np
from heapq import heapify, heappop


# @njit
# def sqrt_sort_heap(V: np.ndarray):
#     n = V.size

#     # Divide em sqrt(n) partes
#     partes = particiona_array(V)

#     # Transforma cada parte em uma heap
#     partes = [TrackedHeap(parte.V, idx) for idx, parte in enumerate(partes)]

#     # Mapeia o valor do maior elemento de cada parte e o index da parte de origem
#     tracked_heap = []
#     maiores_valores, indices_de_origem = [], []
#     for parte in partes:
#         index_da_parte, maior_valor = parte.extract_max()

#         maiores_valores.append(maior_valor)
#         indices_de_origem.append(index_da_parte)

#     # # Aloca o vetor de solução
#     # solucao = np.zeros(n, dtype=np.int64)

#     # # Transforma esses valores mapeados em uma heap auxiliar
#     # max_partes_heap = TrackedHeap(maiores_valores, indices_de_origem)

#     # for i in range(n):
#     #     # Retira o maior item da heap auxiliar
#     #     idx_parte_de_origem, max_value = max_partes_heap.extract_max()

#     #     # Adiciona o valor do item na solucao
#     #     solucao[i] = max_value

#     #     parte = partes[idx_parte_de_origem]  # Parte de onde o elemento foi retirado
#     #     # Se a parte não estiver vazia, adiciona o maior elemento dela na heap auxiliar
#     #     if parte.size >= 1:
#     #         max_partes_heap.insert(*parte.extract_max())

#     # return solucao[::-1]


@njit
def sqrt_sort_heap(V: np.ndarray):
    n = V.size

    partes = [
        Heap([Node(valor, idx) for valor in parte.V])
        for idx, parte in enumerate(particiona_array(V))
    ]
    for parte in partes:
        heapify(parte.values)

    # Transforma cada parte em uma heap
    # partes = [TrackedHeap(parte.V, idx) for idx, parte in enumerate(partes)]

    # Mapeia o valor do maior elemento de cada parte e o index da parte de origem
    # tracked_heap = []
    # maiores_valores, indices_de_origem = [], []
    # for parte in partes:
    #     index_da_parte, maior_valor = parte.extract_max()

    #     maiores_valores.append(maior_valor)
    #     indices_de_origem.append(index_da_parte)

    # # Aloca o vetor de solução
    # solucao = np.zeros(n, dtype=np.int64)

    # # Transforma esses valores mapeados em uma heap auxiliar
    # max_partes_heap = TrackedHeap(maiores_valores, indices_de_origem)

    # for i in range(n):
    #     # Retira o maior item da heap auxiliar
    #     idx_parte_de_origem, max_value = max_partes_heap.extract_max()

    #     # Adiciona o valor do item na solucao
    #     solucao[i] = max_value

    #     parte = partes[idx_parte_de_origem]  # Parte de onde o elemento foi retirado
    #     # Se a parte não estiver vazia, adiciona o maior elemento dela na heap auxiliar
    #     if parte.size >= 1:
    #         max_partes_heap.insert(*parte.extract_max())

    # return solucao[::-1]


# |────────────────────────────────────────────────────────────────────────────────────────────────────────────────────|


Item = namedtuple("Item", ["idx_parte", "item"])


@nb.experimental.jitclass
class Node(object):
    valor: int64
    idx_parte: int64

    def __init__(self, valor, idx_parte):
        self.valor = valor
        self.idx_parte = idx_parte


@nb.experimental.jitclass
class Heap(object):
    heap: List[Node]
    size: int64

    def __init__(self, heap: List[Node]):
        self.heap = nb.typed.List(heap)
        self.size = len(heap)

    def make_max_heap(self):
        for i in range((self.size) // 2, -1, -1):
            heapify_down(self, i)

    def extract_max(self):
        heap = self

        largest = heap[0]

        last = heap[-1]
        heap.size -= 1

        if heap.size >= 1:
            heap[0] = last  # Coloca o ultimo elemento no lugar do maior
            heapify_down(heap, 0)

        return largest

    def heapify_down(self, i):
        heapify_down(self, i)

    def heapify_up(self, i):
        heapify_up(self, i)

    def insert(self, item, index):
        insert(self, item, index)


@njit
def heapify_down(heap: Heap, i):
    maior = i
    l = (i * 2) + 1
    r = (i * 2) + 2

    if (l < len(heap)) and (heap[l] > heap[maior]):
        maior = l

    if (r < len(heap)) and (heap[r] > heap[maior]):
        maior = r

    if maior != i:
        heap[i], heap[maior] = heap[maior], heap[i]  # troca
        heapify_down(heap, maior)


@njit
def heapify_up(heap, i):
    pai = (i - 1) // 2

    if (pai >= 0) and (heap[i].item > heap[pai].item):
        heap[pai], heap[i] = heap[i], heap[pai]  # troca
        heapify_up(heap, pai)


@njit
def insert(heap: Heap, index, item):
    heap.size += 1
    heap[-1] = index, item

    heapify_up(heap, heap.size - 1)


#     def _make_max_heap(self):
#         for i in range((self.size) // 2, -1, -1):
#             heapify_down(self, i)

#     def extract_max(self):
#         heap = self

#         largest = heap[0]

#         last = heap[-1]
#         heap.size -= 1

#         if heap.size >= 1:
#             heap[0] = last  # Coloca o ultimo elemento no lugar do maior
#             heapify_down(heap, 0)

#         return largest

#     def heapify_down(self, i):
#         heapify_down(self, i)

#     def heapify_up(self, i):
#         heapify_up(self, i)

#     def insert(self, item, index):
#         insert(self, item, index)


# @njit
# def heapify_down(heap: TrackedHeap, i):
#     maior = i
#     filho_esq = (i * 2) + 1

#     filho_dir = (i * 2) + 2

#     if (filho_esq < heap.size) and (heap[filho_esq].item > heap[maior].item):
#         maior = filho_esq

#     if (filho_dir < heap.size) and (heap[filho_dir].item > heap[maior].item):
#         maior = filho_dir

#     if maior != i:
#         heap[i], heap[maior] = heap[maior], heap[i]  # troca
#         heapify_down(heap, maior)


# @njit
# def heapify_up(heap, i):
#     pai = (i - 1) // 2

#     if (pai >= 0) and (heap[i].item > heap[pai].item):
#         heap[pai], heap[i] = heap[i], heap[pai]  # troca
#         heapify_up(heap, pai)


# @njit
# def insert(heap: TrackedHeap, index, item):
#     heap.size += 1
#     heap[-1] = index, item

#     heapify_up(heap, heap.size - 1)


# |────────────────────────────────────| Versão simplificado para por no projeto |────────────────────────────────────|
# from heapq import heapify, heappop, heappush

# TrackedValue = namedtuple("TrackedValue", ["value", "idx_parte"])

# def sqrt_sort_heap(V):
#     n = len(V)
#     # Divide em ceil(sqrt(n)) partes
#     partes = particiona_array(V)

#     for parte in partes:
#         heapify(parte)

#     # heap auxiliar que guarda o menor elemento retirado de cada parte e o index da parte correspondente
#     tracked_heap = []
#     for idx_parte in range(len(partes)):
#         value = heappop(partes[idx_parte])

#         tracked_value = TrackedValue(value, idx_parte)
#         tracked_heap.append(tracked_value)
#     heapify(tracked_heap)

#     solucao = []
#     for _ in range(n):
#         # Retira o menor item da heap auxiliar
#         tracked_value = heappop(tracked_heap)

#         # Adiciona o valor do item na solucao
#         solucao.append(tracked_value.value)

#         # Atualiza a heap auxiliar com o próximo menor elemento da parte de origem
#         parte_origem = partes[tracked_value.idx_parte]
#         if len(parte_origem) != 0:
#             # Retira o menor elemento da parte
#             value = heappop(parte_origem)

#             # Adiciona o novo valor rastreado na heap auxiliar
#             new_tracked_value = TrackedValue(value, tracked_value.idx_parte)
#             heappush(tracked_heap, new_tracked_value)
#     return solucao

sqrt_sort_heap(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

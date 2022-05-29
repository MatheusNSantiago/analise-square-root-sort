import numba as nb
import numpy as np

from collections import namedtuple

from common import particiona_array

Item = namedtuple("Item", ["idx_parte", "item"])


@nb.experimental.jitclass([("heap", nb.int64[:, :]), ("size", nb.int64)])
class Heap:
    def __init__(self, V, idx_parte=None):
        self.heap = np.zeros((2, len(V)), dtype=np.int64)

        if idx_parte != None:
            self.heap[0, :] = idx_parte

        self.heap[1, :] = V
        self.size = len(V)

        self._make_max_heap()

    def __getitem__(self, index) -> Item:
        if index >= self.size:
            raise IndexError("Index fora da range")
        if index < 0:
            index = self.size + index

        return Item(self.heap[0][index], self.heap[1][index])

    def __setitem__(self, index, new_value):
        if index < 0:
            index = self.size + index

        self.heap[:, index] = new_value

    def _make_max_heap(self):
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
        

@nb.njit
def heapify_down(heap: Heap, i):
    # Mapeia o heap para uma binary tree
    maior = i
    filho_esq = (i * 2) + 1
    filho_dir = (i * 2) + 2

    if (filho_esq < heap.size) and (heap[filho_esq].item > heap[maior].item):
        maior = filho_esq

    if (filho_dir < heap.size) and (heap[filho_dir].item > heap[maior].item):
        maior = filho_dir

    if maior != i:
        heap[i], heap[maior] = heap[maior], heap[i]  # troca
        heapify_down(heap, maior)


@nb.njit
def heapify_up(heap, i):
    pai = (i - 1) // 2

    if (pai >= 0) and (heap[i].item > heap[pai].item):
        heap[pai], heap[i] = heap[i], heap[pai]  # troca
        heapify_up(heap, pai)

@nb.njit
def insert(heap: Heap, index, item):
    heap.size += 1
    heap[-1] = index, item

    heapify_up(heap, heap.size - 1)

@nb.njit
def sqrt_sort_heap(V: np.ndarray):
    n = V.size

    # Divide em sqrt(n) partes
    partes = particiona_array(V)

    # Transforma cada parte em uma heap
    partes = [Heap(parte.V, idx) for idx, parte in enumerate(partes)]

    # Aloca o vetor de solução
    solucao = np.zeros(n, dtype=np.int64)

    # Mapeia o valor do maior elemento de cada parte e o index da parte de origem
    maiores_valores, indices_de_origem = [], []
    for parte in partes:
        index_da_parte, maior_valor = parte.extract_max()

        maiores_valores.append(maior_valor)
        indices_de_origem.append(index_da_parte)

    # Transforma esses valores mapeados em uma heap auxiliar
    max_partes_heap = Heap(maiores_valores, indices_de_origem)

    for i in range(n):
        # Retira o maior item da heap auxiliar
        idx_parte_de_origem, max_value = max_partes_heap.extract_max()

        # Adiciona o valor do item na solucao
        solucao[i] = max_value

        parte = partes[idx_parte_de_origem]  # Parte de onde o elemento foi retirado
        # Se a parte não estiver vazia, adiciona o maior elemento dela na heap auxiliar
        if parte.size >= 1:
            max_partes_heap.insert(*parte.extract_max())

    return solucao[::-1]


from time import time
import numpy as np
from common import progress_bar
from heap import sqrt_sort_heap
from quadratico import sqrt_sort_quadratico


sizes = [10**4, 10**5, 10**6, 10**7]
repeticoes = 30  # Quantas vezes ele vai testar cada tamanho


res_quadratico = []
res_heap = []
metodos = [sqrt_sort_quadratico, sqrt_sort_heap]


for i, metodo in enumerate(metodos):

    print("Pelo Método Quadratico" if i == 0 else "Usando Heap")
    for size in sizes:
        tempos = []
        for _ in progress_bar(repeticoes, desc=f"  Para n = 10^{int(np.log10(size))}"):
            start = time()

            np.random.seed(124)  # Seed the random number generator
            array = np.random.randint(1000, size=size)

            metodo(array)

            end = time()
            tempos.append(end - start)

        media = round(np.mean(tempos), 3)
        if i == 0:
            res_quadratico.append(media)
        else:
            res_heap.append(media)


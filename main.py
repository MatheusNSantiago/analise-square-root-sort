from time import time
import numpy as np
from utils import progress_bar, rprint, show_results_table
from quadratico import sqrt_sort_quadratico
from heap import sqrt_sort_heap
from common import plot_array

if __name__ == "__main__":
    sizes = [
        # 10**1,
        # 10**2,
        # 10**3,
        10**4,
        10**5,
        10**6,
        10**7,
    ]
    repeticoes = 30  # Quantas vezes ele vai testar cada tamanho

    resultados = {
        "quadratico": {size: [] for size in sizes},
        "heap": {size: [] for size in sizes},
    }   

    for i, metodo in enumerate(
        [
            sqrt_sort_quadratico,
            sqrt_sort_heap,
        ]
    ):
        rprint("Usando Heap" if i else "Pelo Método Quadratico", style="bold underline")
        for size in sizes:
            tempos = []
            for _ in progress_bar(repeticoes, size):
                start = time()

                array = np.random.randint(1000, size=size)
                metodo(array)


                end = time()
                tempos.append(end - start)

            if i == 0:
                resultados["quadratico"][size] = tempos
            else:
                resultados["heap"][size] = tempos

    plot_array(resultados, sizes)
    
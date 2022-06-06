from re import M
import numba as nb
import numpy as np
from common import Parte, particiona_array

VAZIO = np.iinfo(np.int64).min  # equivalente a -infinito



# @nb.njit
# def sqrt_sort_quadratico(V: np.ndarray):
#     n = len(V)

#     # Divide em sqrt(n) partes
#     partes: list[Parte] = particiona_array(V)

#     # Ordena cada parte usando o insertion sort
#     for parte in partes:
#         insertion_sort(parte)

#     solucao = np.zeros(n, dtype=np.int64)

#     # vetor com o maior elemento de cada parte
#     max_partes = np.array([parte[-1] for parte in partes])
#     for i in range(n):
#         # pega o index do maior elemento entre os maiores
#         max_index = np.argmax(max_partes)

#         # parte que contem o maior elemento
#         parte = partes[max_index]

#         # Adiciona esse elemento na solução e remove ele do vetor
#         solucao[i] = parte.pop()

#         # Atualiza o maior elemento da parte retirada. Se a parte ficou vazia, sinaliza com VAZIO
#         max_partes[max_index] = parte[-1] if parte.size != 0 else VAZIO

#     return solucao[::-1]

def sqrt_sort_quadratico(V):
    # Divide $V$ em $\lceil\; n/\sqrt{n}\; \rceil$ partes
    partes = particiona_array(V)

    # Ordena cada parte usando o insertion sort
    for parte in partes:
        insertion_sort(parte)
        # print(parte)

    solucao = []

    # vetor auxiliar representando o maior elemento de cada parte
    max_partes = [parte[-1] for parte in partes]
    
    for _ in range(len(V)):
        # pega o index do maior elemento entre os maiores
        max_index = np.argmax(max_partes)

        # parte que contem o maior elemento
        parte = partes[max_index]

        # remove o maior elemento da parte de origem e o adiciona na solucao
        maior_valor = parte.pop()
        solucao.append(maior_valor)

        # Repõe o maior elemento da parte retirada. Se a parte ficou vazia, seta o maior elemento como $-\infty$ para que essa parte seja ignorada futuramente 
        if len(parte) != 0:
            max_partes[max_index] =  parte[-1]
        else:
            max_partes[max_index] =  -np.inf

    return solucao

# @nb.njit
def insertion_sort(parte: Parte):
    # for j in range(1, parte.size):
    for j in range(1, len(parte)):
        key = parte[j]
        i = j - 1
        while (i >= 0) and (parte[i] > key):
            parte[i + 1] = parte[i]
            parte[i] = key
            i -= 1

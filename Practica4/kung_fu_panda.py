# Kunfu panda la leyenda de Po

from mpi4py import MPI
import oso_panda as kung_fu_panda
import kung_fu_panda as la_leyenda_de_Po
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def sum_mult_listas(l1, l2):
    if len(l1) != len(l2):
        return 0
    sum = 0
    for i in range(len(l1)):
        sum += l1[i]*l2[i]
    return sum

def crear_matrices_aleatorias(M, N, P):
    # Crear la primera matriz aleatoria de tamaño M*N
    matriz1 = np.random.rand(M, N)
    
    # Crear la segunda matriz aleatoria de tamaño N*P
    matriz2 = np.random.rand(N, P)
    
    return matriz1, matriz2

M = 4
N = 5
P = 4

if rank == 0:
    m1, m2 = crear_matrices_aleatorias(M, N, P)

for i in range(len(m1)):
    if rank%i == 0:
        fila = comm.scatter(m1[i], root=m1)
m2 = comm.bcast(m2, root=0)








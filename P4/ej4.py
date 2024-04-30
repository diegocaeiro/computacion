# Kunfu panda la leyenda de Po

from mpi4py import MPI
import numpy as np
from time import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M = 8001
N = 4000
P = 4000


if rank == 0:
    m1 = np.random.rand(M, N)
    m2 = np.random.rand(N, P)
    m1_simply = m1
    rest = 0

else: 
    m1_simply = np.empty((M, N))
    m2 = np.empty((N, P))



# Resolución distribuida de la multiplicación de matrices

if rank == 0:
    tiempo_dist = time()


if M % size != 0:
    if rank == 0: 
        rest = M % size
        row_rest = m1[-rest:] # Filas restantes
        m1_simply = m1[:-rest]
        comm.send(row_rest, dest=size-1) # Enviar las filas restantes al último proceso
        
    else:
        m1_simply = np.empty((M, N))
        m2 = np.empty((N, P))

    if rank == size-1: # El último proceso recibe las filas restantes
        row_rest = comm.recv(source=0)


# if rank == 0:
#    print(m1.shape)


# Envío de datos a los procesos

local_m1 = np.empty((M // size, N))

time_scatter = time()

comm.Scatter(m1_simply, local_m1, root=0)
comm.Bcast(m2, root=0)

time_scatter = time() - time_scatter


# if rank == size-1:
#     local_m1 = np.vstack((local_m1, row_rest))



# Multiplicación de matrices en cada proceso

if rank == 0:
    time_dot = time()

# print('rank:', rank, 'local_m1:', local_m1.shape, 'm2:', m2.shape)
result = np.dot(local_m1, m2)

if rank == 0:
    time_dot = time() - time_dot


# Recopilación de datos en el proceso 0

time_gather = time()

gathered_data = None

if rank == 0:
    gathered_data = np.zeros((M-rest, P))

comm.Gather(result, gathered_data, root=0)  

time_gather = time() - time_gather


# Añadir filas restantes al proceso 0 (las extra)

time_extra = time()

if M % size != 0:

    if rank == size-1:
        result_extra_rows = np.dot(row_rest, m2)
        comm.send(result_extra_rows, dest=0)

    if rank == 0:
        # print('Matriz gathered_inicial:', gathered_data)
        extra_rows = comm.recv(source=size-1)
        gathered_data = np.vstack((gathered_data, extra_rows))
        # print('Matriz resultante:', gathered_data)


time_extra = time() - time_extra

if rank == 0:
    tiempo_dist = time() - tiempo_dist

if rank == 0:
    print('fase scatter:', time_scatter)
    print('fase dot:', time_dot)
    print('fase gather:', time_gather)
    print('fase extra:', time_extra)

# Resolución secuencial de la multiplicación de matrices

if rank == 0:
    tiempo_sec = time()

    result_sec = np.dot(m1, m2)

    tiempo_sec = time() - tiempo_sec


# Comprobación y comparación de resultados

if rank == 0:
    if np.allclose(gathered_data, result_sec):
        print('Las matrices son iguales')
        print('tiempo_sec:', tiempo_sec)
        print('tiempo_dist:', tiempo_dist)
        print('Speedup:', tiempo_sec/tiempo_dist)
    else:
        print('Las matrices son diferentes')

MPI.Finalize() # Finalizar MPI





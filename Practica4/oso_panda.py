from mpi4py import MPI
import random
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1000
R = 8
contador = 0
verification = True

while verification and contador < N:
    verification = False

    n = random.randint(0,R)
    print(rank,'genera',n)
    sys.stdout.flush()

    for i in range(size):

        data = comm.bcast(n, root=i)
        print(f'{rank} recibe {data} de {i}')
        sys.stdout.flush()

        if data != n:
            verification = True
    
    contador += 1

if rank == 0:
    print(f'Se hicieron {contador} ejecuciones del juego.')


if contador < N:
    print(f'Todos los procesos sacaron el número {n} (esta verificación pertenece al proceso {rank})')

else:
    print('No sacamos el mismo número :(')


    
        
            
        
    








































































































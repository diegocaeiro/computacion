from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    goal = 20
    data = random.randint(1,5)
    for i in range(1,size):
        comm.send(goal, dest=i)
    
else:
    goal = comm.recv(source=0)
    print(f'Proceso {rank} recibió: {goal}')

if rank == 0:
    comm.send(data,dest=rank+1)

while True:
    if rank != 0: 
        data = comm.recv(source=rank-1) 
    else: 
        data = comm.recv(source=size-1)
        
    if data >= goal:

        if rank != size-1: 
            comm.send(data,dest=rank+1) 
        else: 
            comm.send(data,dest=0)
            
        break

    print(f'El valor recibido por el proceso {rank} fue {data}')
    
    random2 = random.randint(1,5)
    data = data + random2

    print(f'El proceso {rank} aumenta el valor recibido en {random2}')

    if rank != size-1: 
        comm.send(data,dest=rank+1) 
    else: 
        comm.send(data,dest=0)

print(f'Print final del proceso {rank} -> {data}')
MPI.Finalize()
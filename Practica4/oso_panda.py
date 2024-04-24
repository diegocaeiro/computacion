from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


R = 42
n = random.randint(0,R)

print(rank,'genera',n)

data = comm.bcast(n, root=0)
data1 = comm.bcast(n, root=1)
data2 = comm.bcast(n, root=2)
data3 = comm.bcast(n, root=3)


print(rank,'recive',data)

if rank != 0:
    print("Process", rank, "received data:", data)


    
    








































































































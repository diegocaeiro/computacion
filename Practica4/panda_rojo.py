from mpi4py import MPI
import random
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cart_comm = comm.Create_cart(dims=[2, 2, 2], periods=[1, 1, 1], reorder=True)

coords = cart_comm.Get_coords(rank)

recv_left, recv_right = cart_comm.Shift(0, 1) # (hacia qué coordenada te mueves, cuantas unidades te mueves)
recv_up, recv_down = cart_comm.Shift(1, 1)
recv_front, recv_back = cart_comm.Shift(2, 1)


print(f'El nodo {rank} tiene las coordenadas ({coords[0]}, {coords[1]}, {coords[2]}) ')
print(recv_up+recv_down+recv_front+recv_back+recv_right+recv_left)
# Montecarlo

from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import math
import time

# Datos del mundo con Excepciones para secuencial
def create_data_world (num_points, s = False):

    if s:
        comm = None
        size = 1
        rank = 0

    else: 
        try:
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            num_points = int(math.floor(num_points/size))

        except:
            comm = None
            size = 1
            rank = 0

    return comm, size, rank, num_points

# Generar la lista de tuplas aleatorias
def random_list(num_points):
    
    points = np.array([(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(num_points)]) # , dtype='float64'

    return points

# Crear espacio para los datos (en el caso de Gather)
def gathering_array (num_points, size):
    
    points_a = np.zeros(num_points*size) # , dtype='float64'

    return points_a

# Reunir los datos
def gather(points, comm, points_a = None):
    
    try:
        points_a = comm.gather(points, root=0)
        # comm.Gather(points, points_a) # El Gather que no funciona

    except:
        points_a = points

    return points_a

# Separar los puntos dentro y fuera del círculo
def inside_outside (points_a):
    
    inside_circle = []
    outside_circle = []
    for point in points_a:
        if np.sqrt(point[0]**2 + point[1]**2) <= 1:
            inside_circle.append(point)
        else:
            outside_circle.append(point)

    inside_circle = np.array(inside_circle)
    outside_circle = np.array(outside_circle)

    return inside_circle, outside_circle

#
def data (inside_circle,outside_circle):
    num_inside_a = len(inside_circle)
    num_outside_a = len(outside_circle)
    pi_aprox = num_inside_a/(num_inside_a+num_outside_a)*4
    return num_inside_a, num_outside_a, pi_aprox

# Gráfica y datos
def graph_and_data (inside_circle, outside_circle): 
    
    num_inside_a, num_outside_a, pi_aprox = data(inside_circle, outside_circle)

    plt.scatter(outside_circle[:,0], outside_circle[:,1], color='red', label=f'{num_inside_a} Fuera del círculo')
    plt.scatter(inside_circle[:,0], inside_circle[:,1], color='blue', label=f'{num_outside_a} Dentro del círculo')

    circle = plt.Circle((0, 0), 1, color='green', fill=False)
    plt.gca().add_patch(circle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Puntos dentro y fuera del círculo')
    plt.legend()
    print(f'\nDe {num_inside_a+num_outside_a} puntos, cayeron \n{num_inside_a} dentro y \n{num_outside_a} fuera\nAproximación de pi = {pi_aprox} \nDiferencia con pi {pi-pi_aprox}')
    plt.show()

def graph_error (pi_errors,l):
    # l = np.array(l)
    # pi_errors = np.array(pi_errors)

    plt.plot(l, pi_errors, marker='o', linestyle='-')
    plt.xlabel('Número de puntos')
    plt.ylabel('Error')
    plt.title('Diferencia con Pi según aumenta el número de puntos')
    plt.grid(True)
    plt.show()


# Speed up

num_points = 100000

if MPI.COMM_WORLD.Get_rank() == 0:
    # Secuencial
    t_sec = time.time()

    size = 1
    rank = 0
    points_a = random_list(num_points)
    inside_circle, outside_circle = inside_outside(points_a)

    t_sec = time.time() - t_sec

    # Paralelo
    t_par = time.time()

comm, size, rank, num_points = create_data_world (num_points)
points = random_list(num_points)

points_a = gather(points, comm) # Alomejor hay que quitar el np.concatenate, porque alomejor el concatenate solo se puede hacer en rank == 0 #del

if rank == 0:
    points_a = np.concatenate(points_a) # En efecto, hay que hacer el concatenate aquí, porque el los rank != 0 da error por el None que tienen en points_a
    inside_circle, outside_circle = inside_outside(points_a)
    t_par = time.time() - t_par
    print(f'El tiempo del secuencial fue de {t_sec} s y el del paralelo {t_par}, para un speed up de {t_sec/t_par}')
    graph_and_data (inside_circle, outside_circle)


# Error

pi_errors = []
l = [100, 1000, 10000, 100000, 1000000]
for n in l:
    num_points = n
    comm, size, rank, num_points = create_data_world (num_points)
    points = random_list(num_points)

    points_a = gather(points, comm) # Alomejor hay que quitar el np.concatenate, porque alomejor el concatenate solo se puede hacer en rank == 0 #del

    if rank == 0:
        points_a = np.concatenate(points_a) # En efecto, hay que hacer el concatenate aquí, porque el los rank != 0 da error por el None que tienen en points_a
        inside_circle, outside_circle = inside_outside(points_a)
        num_inside_a, num_outside_a, pi_aprox = data(inside_circle, outside_circle)
        pi_errors.append(abs(pi_aprox-pi))

if rank == 0:
    graph_error(pi_errors,l)

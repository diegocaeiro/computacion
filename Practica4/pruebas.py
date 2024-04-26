import numpy as np

def crear_matrices_aleatorias(M, N, P):
    # Crear la primera matriz aleatoria de tamaño M*N
    matriz1 = np.random.rand(M, N)
    
    # Crear la segunda matriz aleatoria de tamaño N*P
    matriz2 = np.random.rand(N, P)
    
    return matriz1, matriz2



print(crear_matrices_aleatorias(4,4,4))

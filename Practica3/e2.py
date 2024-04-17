import multiprocessing.managers
import threading
import multiprocessing
import random
import time

n_lineas = int(input('Introduce el número de líneas del archivo >>> '))
n = int(input('Introduce el número de hilos o procesos >>> '))



"""**************************HILOS**************************"""

print('\nHILOS\n')

ficherinho = ''
lista_contadores = [0]*n

lock = threading.Lock()

def generar_datos(n_lineas, n):
        global ficherinho
        ficherinho = open ('ficherinho.txt', 'w') # Abrimos el fichero para poder escribir en el ('w')
        guardado = ''
        for _ in range(n_lineas):
            guardado +=  str(random.randint(1,n)) + '\n'
        
        ficherinho.write (guardado) #Escribimos el string en el documento y lo cerramos
        ficherinho.close()

def contante(numero_hilo):
    global lista_contadores
    contador = 0
    archivo_lectura = open('ficherinho.txt', 'r')
    lineas = archivo_lectura.readlines()
    for linea in lineas:
        # print(f'Hilo: {numero_hilo} linea:{linea}')
        if int(linea[:-1]) == numero_hilo:
            contador += 1
    print(f'Hilo {numero_hilo}: {contador}')
    with lock:
        lista_contadores[numero_hilo - 1] = contador
    


# Crear generador
generador = threading.Thread(target = generar_datos, args=(n_lineas,n))

# Tiempo generador
tiempo_generador = time.time()
generador.start()
generador.join()
tiempo_generador = time.time() - tiempo_generador

print('El tiempo requerido por el nodo generador ha sido de: {}'.format(tiempo_generador))


# Crear e inciar contadores
contadores = []

tiempo_necesario_h = time.time()
for i in range(n):
    contador = threading.Thread(target = contante, args=(i+1, ))
    contadores.append(contador)
    contador.start()

for h in contadores:
    h.join()


lista = lista_contadores.copy()

# Encontrar el valor máximo en la lista
max_valor = max(lista)

# Obtener los índices donde aparece el valor máximo
lista_max = [i+1 for i, valor in enumerate(lista) if valor == max_valor]
tiempo_necesario_h = time.time() - tiempo_necesario_h


print('\n\nEl tiempo necesario para realizar el conteo ha sido de >>> {}'.format(tiempo_necesario_h))

# lista_max = [(lista.index(i)+1) for i in lista if i == max(lista)]
print('El número que más veces se ha repetido ha(n) sido {} con un total de {} veces'.format(lista_max, max_valor))


"""**************************PROCESOS**************************"""

print('\nPROCESOS\n')


manager = multiprocessing.Manager()
lista_contadores = manager.list([0]*n)
ficherinho = ''


def contante_p(numero_proceso):
    global lista_contadores
    contador = 0
    archivo_lectura = open('ficherinho.txt', 'r')
    lineas = archivo_lectura.readlines()
    for linea in lineas:
        # print(f'Hilo: {numero_hilo} linea:{linea}')
        if int(linea[:-1]) == numero_proceso:
            contador += 1
    print(f'Proceso {numero_proceso}: {contador}')
    lista_contadores[numero_proceso - 1] = contador


# Crear generador
generador = multiprocessing.Process(target = generar_datos, args=(n_lineas,n))

# Tiempo generador
tiempo_generador = time.time()
generador.start()
generador.join()
tiempo_generador = time.time() - tiempo_generador

print('El tiempo requerido por el proceso generador ha sido de: {}'.format(tiempo_generador))


# Crear e inciar contadores
contadores_p = []

tiempo_necesario_p = time.time()
for i in range(n):
    contador = multiprocessing.Process(target = contante_p, args=(i+1,))
    contadores_p.append(contador)
    contador.start()

for p in contadores_p:
    p.join()

print(f'Lista de contadores>> {lista_contadores}')
lista = lista_contadores[:]

# Encontrar el valor máximo en la lista
max_valor = max(lista)

# Obtener los índices donde aparece el valor máximo
lista_max = [i+1 for i, valor in enumerate(lista) if valor == max_valor]
tiempo_necesario_p = time.time() - tiempo_necesario_p

print('\n\nEl tiempo necesario para realizar el conteo ha sido de >>> {}'.format(tiempo_necesario_p))

print('El número que más veces se ha repetido ha(n) sido {} con un total de {} veces'.format(lista_max, max_valor))


"""**************************ANÁLISIS**************************"""

# tiempos = [100,1000,10000,100000,1000000,10000000]
# procesadores = [1,2,4,8]

# for procesador in procesadores:
#      for tiempo in tiempos:
          

# Calcular speedup 
speedup = tiempo_necesario_h / tiempo_necesario_p
print("Speedup:", speedup)
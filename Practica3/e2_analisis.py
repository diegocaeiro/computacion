import multiprocessing.managers
import threading
import multiprocessing
import random
import time


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
    with lock:
        lista_contadores[numero_hilo - 1] = contador
    

def contante_p(numero_proceso):
    global lista_contadores
    contador = 0
    archivo_lectura = open('ficherinho.txt', 'r')
    lineas = archivo_lectura.readlines()
    for linea in lineas:
        # print(f'Hilo: {numero_hilo} linea:{linea}')
        if int(linea[:-1]) == numero_proceso:
            contador += 1
    lista_contadores[numero_proceso - 1] = contador


# PROGRAMA

lineas = [100000, 1000000, 10000000]
numeros = [1,2,4,8]


for n in numeros:
    print(f'\nNúmero de hilos/procesos: {n}')
    for n_lineas in lineas:
        print(f'\n\tNúmero de lineas {n_lineas}')
        ficherinho = ''
        lista_contadores = [0]*n
        lock = threading.Lock()

        """**************************HILOS**************************"""

        # Crear generador
        generador = threading.Thread(target = generar_datos, args=(n_lineas,n))

        # Tiempo generador
        tiempo_generador = time.time()
        generador.start()
        generador.join()
        tiempo_generador = time.time() - tiempo_generador

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

        """**************************PROCESOS**************************"""

        manager = multiprocessing.Manager()
        lista_contadores = manager.list([0]*n)
        generador = multiprocessing.Process(target = generar_datos, args=(n_lineas,n))

        # Tiempo generador
        tiempo_generador = time.time()
        generador.start()
        generador.join()
        tiempo_generador = time.time() - tiempo_generador

        # Crear e inciar contadores
        contadores_p = []

        tiempo_necesario_p = time.time()

        for i in range(n):
            contador = multiprocessing.Process(target = contante_p, args=(i+1,))
            contadores_p.append(contador)
            contador.start()

        for p in contadores_p:
            p.join()

        lista = lista_contadores[:]

        # Encontrar el valor máximo en la lista
        max_valor = max(lista)

        # Obtener los índices donde aparece el valor máximo
        lista_max = [i+1 for i, valor in enumerate(lista) if valor == max_valor]
        tiempo_necesario_p = time.time() - tiempo_necesario_p

        speedup = tiempo_necesario_h / tiempo_necesario_p

        print(f'\n \t\tTIEMPOS: \n Tiempo hilos: {tiempo_necesario_h} \n Tiempo procesos: {tiempo_necesario_p} \n Speedup: {speedup}')
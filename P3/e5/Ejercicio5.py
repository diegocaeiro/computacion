from ELM_base import *


"""*****************************IMPORTACIÓN DE LOS DATOS*****************************"""
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


iris = load_iris()
digits = load_digits()
wine = load_wine()
breast_cancer = load_breast_cancer()

# input_size tiene tantos elementos como datos cada individuo de la especie
# hidden_size es el número de neuronas por el medio (+ = +preciso y más costoso)
# output_size es 1 pq estamos en un sistema de clasificación

ia = ELMMP(4,10,1)

X = iris.data
y = iris.target

X = np.tile(X, (100, 1)) # Datos de la clase iris
y = np.tile(y, (100,1)).flatten()#) lo que corresponde a cada dato


porcentaje_test = 80

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= porcentaje_test/100, random_state=42)

encoder = OneHotEncoder() 
print(X_test) # Son 120 flores pq he puesto qu euse el 80%
print(y_train)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
print(y_train) # en la posición de primer argumento de la tupla quiero que salga el segundo argumento de la tupla y que esté al % que indica el tercer elemento


ia.train(X_train, y_train)
prediccion = ia.predict(X_test) # Tiene tantos elementos como los que tiene X_test
print(prediccion)
print(y_test)

# Una función que me diga la precisión global comparando y_test con X_test
def funcion_precision_global(entrenar, y_test):
    precision = len(y_test)
    for indice,i in enumerate(y_test):
        if i != entrenar[indice]:
            precision -= 1
    return precision/len(y_test), len(y_test) - precision


# Una función que me diga la precisión por elemento
def funcion_prediccion_individual(entrenar, y_test):
    elementos, conteos_inicial = np.unique(y_test, return_counts=True)
    conteos = conteos_inicial.copy()
    for indice, i in enumerate(y_test):
        if i != entrenar[indice]:
            index = np.where(elementos == i)[0][0]
            conteos[index] -= 1
    precisiones = list()
    for indice,i in enumerate(conteos_inicial):
        precisiones.append(conteos[indice] / i)
    return elementos, conteos_inicial, conteos, precisiones



print(f'La precisión de la predicción ha sido de: {(funcion_precision_global(prediccion, y_test) * 100)[0]}% con {(funcion_precision_global(prediccion, y_test) * 100)[1]} errores')


print(f"""La lista de elementos es: {funcion_prediccion_individual(prediccion, y_test)[0]},
la lista real de conteo de elementos es: {funcion_prediccion_individual(prediccion, y_test)[1]},
la lista predictiva de conteo de elementos es: {funcion_prediccion_individual(prediccion, y_test)[2]},
la precisión de cada elemento ha sido de: {[str(i * 100)+'%' for i in funcion_prediccion_individual(prediccion, y_test)[3]]}""")



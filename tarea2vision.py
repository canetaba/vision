#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
import numpy as np
import sys
import ruta as _ruta
sys.path.insert(0, _ruta.cnn_lib_location)
import cnnLib.deep_searcher as cnn
from distancia import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt



# Datos de train
labels_train = './train/labels.npy'
ruta_train = './train/features.des'
names_train = './train/names.txt'

# Datos de test
labels_test = './test/labels.npy'
ruta_test = './test/features.des'
names_test = './test/names.txt'

# Distancia Euclidiana
dist_eu = np.array([])
etiqueta_base = np.array([])


# Redimensiona la matriz de caracteristicas
def redimensionar(arreglo):
    dimension =  int(len(arreglo) / 1024)
    return np.reshape(arreglo, (dimension, 1024))

# Abre la ruta del archivo
def abrir_archivo(ruta_archivo):
    arreglo = np.fromfile(ruta_archivo.format(dir=dir), dtype=np.int32)[3:]
    return arreglo

# Abre las etiquetas
def abrir_etiquetas(ruta_archivo):
    arreglo = np.fromfile(ruta_archivo.format(dir=dir), dtype=np.int32)
    return arreglo

def graficar(vector, etiquetas):
    colors = np.random.rand(6000)
    fig, ax = plt.subplots()
    ax.scatter(vector, etiquetas, c=colors)
    plt.show()

features_train = abrir_archivo(ruta_train)
features_test = abrir_archivo(ruta_test)


features_train = redimensionar(features_train)
features_test = redimensionar(features_test)

etiquetas_train = abrir_etiquetas(labels_train)
# print("Etiquetas TRAIN ", etiquetas_train)

etiquetas_test = abrir_etiquetas(labels_test)
print("Etiquetas TEST ", etiquetas_test)

# Esto se calcula la salida del vector hog, tanto para train como para test
# Normalizacion de los datos
norm_square_train = cnn.norm.square_root_norm(features_train)
norm_square_test = cnn.norm.square_root_norm(features_test)



# Escogemos una imagen no normalizada del conjunto de prueba
NUMERO_IMAGEN = 3
row_a = features_test[NUMERO_IMAGEN]
print("Imagen del conjunto de prueba sin normalizar", row_a)
label_a = etiquetas_test[NUMERO_IMAGEN]
print("Etiqueta del conjunto de prueba sin normalizar", label_a)

# Escogemos una imagen normalizada del conjunto de prueba
rown_a = norm_square_test[NUMERO_IMAGEN]
print("Imagen del conjunto de prueba", rown_a)
label_a = etiquetas_test[NUMERO_IMAGEN]
print("Etiqueta del conjunto de prueba ", label_a)


# TSNE sin normalizar
xtsne = TSNE(n_components=1).fit_transform(features_test)

print("xtsne " , xtsne)
print("etiquetas ", etiquetas_test)
xtsne = np.transpose(xtsne)
graficar(xtsne, etiquetas_test)

# TSNE sin normalizar
xtsne_norm = TSNE(n_components=1).fit_transform(norm_square_test)




# Calcula la distancia euclidiana entre una imagen de prueba y el vector de caracteristicas
# Sin normalizacion
dist_eu = distancia(features_test, row_a)
dist_eu_norm =  distancia(norm_square_test, rown_a)

print(dist_eu)
print(dist_eu_norm)

# Arreglo de etiquetas
# Buscamos el numero total de ground truth
ground_truth = 1
for row_c in etiquetas_test:
   if row_c == int(label_a):
        ground_truth = ground_truth +1
   etiqueta_base = np.append(etiqueta_base, row_c)

# copio el arreglo
aux_euclidiano = np.copy(dist_eu)
aux_euclidiano_norm = np.copy(dist_eu_norm)

# Ordeno de menor a mayor de acuerdo a las distancias
aux_euclidiano.sort()
aux_euclidiano_norm.sort()


# Creo un arreglo de resultados
result = np.array([])

# Escojo con cuantas imagenes quiero comparar la imagen de consulta con las de la BD
numero_imagenes = 20
# Busco las posiciones de las etiquetas de acuerdo a los valores de las distancias
for i in range(0, numero_imagenes):
    donde = np.where(dist_eu == aux_euclidiano[i])
    result = np.append(result, donde)

# Almaceno el valor de las etiquetas en este arreglo
average_precision = 0
contador = 0
positivo = 0

# Busco el contenido de en el arreglo de las etiquetas de acuerdo a los resultados obtenidos anteriormente
# Extraigo el valor de la etiqueta
# Comparo con el valor de la etiqueta de la query con el valor de la etiqueta que encontre
# Si es igual lo cuento y lo sumo de acuerdo al criterio de Average Precision
for row in result:
    contador = contador + 1
    fila = int(row)
    valor_etiqueta = int(etiqueta_base[fila])

    # Si la etiqueta de la consulta es igual a la de la clase
    if int(label_a) == valor_etiqueta:
        positivo = positivo + 1
        average_precision = (positivo / contador) + average_precision

# Imprimo average precision
print("average_precision ", average_precision / numero_imagenes)

# Calculo de mAP
# AP / Nº total de queries
mean_average_precision = average_precision / features_test.size
print("mAP ", mean_average_precision)





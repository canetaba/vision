#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import ruta as _ruta
sys.path.insert(0, _ruta.cnn_lib_location)
import cnnLib.deep_searcher as cnn
from distancia import *


# Datos de train
labels_train = './train/labels.npy'
ruta_train = './train/features.des'
features_train = np.fromfile(ruta_train.format(dir=dir), dtype=np.int32)[3:]


# Datos de test
labels_test = './test/labels.npy'
ruta_test = './test/features.des'
features_test = np.fromfile(ruta_train.format(dir=dir), dtype=np.int32)[3:]

# Esto se calcula la salida del vector hog, tanto para train como para test
# Normalizacion de los datos
normalizacion_square_train = cnn.norm.square_root_norm(features_train)
normalizacion_square_test = cnn.norm.square_root_norm(features_test)

print(features_train)
print(normalizacion_square_train)

print(features_test)
print(normalizacion_square_test)

# Etiquetas
etiquetas_train = np.fromfile(labels_train.format(dir=dir), dtype=np.int32)
print(etiquetas_train)

etiquetas_test = np.fromfile(labels_test.format(dir=dir), dtype=np.int32)
print(etiquetas_test)

# Nombre de las imagenes
names_train= './train/names.txt'
names_test= './test/names.txt'

# Se leen los nombres de las imagenes por linea
with open(names_train.format(dir=dir)) as f:
    names = f.readlines()

# Leemos los nombres de las imagenes
with open(names_test.format(dir=dir)) as f:
    names = f.readlines()

conjunto_train = {'features_train': features_train, 'labels_train': labels_train, 'names_train': names_train}
conjunto_test = {'features_test': features_test, 'labels_test': labels_test, 'names_test': names_test}

# distancia euclidiana
dist_eu = np.array([])
etiqueta_base = np.array([])

# Escogemos una imagen del conjunto de prueba
row_a = features_test[222]
label_a = etiquetas_test[222]


# Calcula la distancia euclidiana entre una imagen de prueba y el vector de caracteristicas
# Con normalizacion
#Imagen de prueba
dist_eu = distancia(features_train, row_a)
print(dist_eu)



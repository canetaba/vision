import numpy as np

"""
    Calcula la distancia Euclidiana entre dos vectores

"""


def distancia(features, row_a):
    distancia_euclidiana = np.array([])
    # Calcula la distancia euclidiana entre dos vectores
    for row_b in features:
        dist = np.linalg.norm(row_a - row_b)
        distancia_euclidiana = np.append(distancia_euclidiana, dist, axis=None)
    return distancia_euclidiana

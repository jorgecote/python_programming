import numpy as np
np.random.seed(5)
from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model
# from keras.optimizers import SGD
# from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
import os
import time

from google.colab import drive
 
drive.mount('/content/drive')   # Se monta el drive para tener disponibles todos los archivos
# Se utiliza la función open (Nativa de python) para abrir el archivo y read para leerlos
nombres = open('/content/drive/MyDrive/Colab Notebooks/TensorFlow Certification Preparation/nombres_dinosaurios.txt','r').read()
nombres = nombres.lower() # Todos los nombres se dejan en minúscula para que no afecten las secuencias de generación
# tam_datos, tam_alfabeto = len(nombres), len(alfabeto)
# nombres=nombres.splitlines()
print(f'El total del texto es: {len(nombres)} caracteres')    # Se calcula el total de caracteres del archivo
print(nombres[:38])                                           # Revisando como luce el conjunto de datos
alfabeto = sorted(set(nombres))                                 # Con la función set python encuentra los caracteres únicos del archivo
print("Total de caracteres: ",len(nombres),", Caracteres únicos: ", len(alfabeto))
print(alfabeto)
chars = tf.strings.unicode_split(nombres, input_encoding='UTF-8')   # Se separan los caracteres y se organizan en un tensor
chars
car_a_ind = tf.keras.layers.StringLookup(vocabulary=alfabeto, mask_token=None)
# { car:ind for ind,car in enumerate(sorted(alfabeto))} # Se genera un diccionario asignando un valor único a cada caracter
car_a_ind(chars)
car_a_ind
print(car_a_ind(chars))
ind_a_car = tf.keras.layers.StringLookup(vocabulary=car_a_ind.get_vocabulary(), invert=True, mask_token=None)
print(ind_a_car(car_a_ind(chars))[:38])
tf.strings.reduce_join(chars, axis=-1).numpy()
def ind_a_texto(ind):           # Función para recuperar cadena de texto desde índices
  return tf.strings.reduce_join(ind_a_car(ind), axis=-1)
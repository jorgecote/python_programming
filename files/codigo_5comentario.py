 import numpy as np
 np.random.seed(5)
 from keras.layers import Input, Dense, SimpleRNN
 from keras.models import Model
 
 
 from keras import backend as K
 import tensorflow as tf
 import os
 import time
 
 from google.colab import drive
  
 drive.mount('/content/drive')   
 
 nombres = open('/content/drive/MyDrive/Colab Notebooks/TensorFlow Certification Preparation/nombres_dinosaurios.txt','r').read()
 nombres = nombres.lower() 
 
 
 print(f'El total del texto es: {len(nombres)} caracteres')    
 print(nombres[:38])                                           
 alfabeto = sorted(set(nombres))                                 
 print("Total de caracteres: ",len(nombres),", Caracteres únicos: ", len(alfabeto))
 print(alfabeto)
 chars = tf.strings.unicode_split(nombres, input_encoding='UTF-8')   
 chars
 car_a_ind = tf.keras.layers.StringLookup(vocabulary=alfabeto, mask_token=None)
 
 car_a_ind(chars)
 car_a_ind
 print(car_a_ind(chars))
 ind_a_car = tf.keras.layers.StringLookup(vocabulary=car_a_ind.get_vocabulary(), invert=True, mask_token=None)
 print(ind_a_car(car_a_ind(chars))[:38])
 tf.strings.reduce_join(chars, axis=-1).numpy()
 def ind_a_texto(ind):           
   return tf.strings.reduce_join(ind_a_car(ind), axis=-1)
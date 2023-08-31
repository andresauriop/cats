import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory

import os
import matplotlib.image as mpimg


path = 'C:/Programas/datasets/dogs-vs-cats/train/'
classes = os.listdir(path) #lista directorios con las clases  train/cats y train/dogs
print(classes) #cats y dogs


fig = plt.gcf()
fig.set_size_inches(10, 10)
cat_dir = 'C:/Programas/datasets/dogs-vs-cats/train/cats/'
dog_dir = 'C:/Programas/datasets/dogs-vs-cats/train/dogs/'
cat_names = os.listdir(cat_dir)  #recupera el nombre de archivos
dog_names = os.listdir(dog_dir)  #recupera el nombre de archivos

pic_index = 210

cat_images = [os.path.join(cat_dir, fname)
    for fname in cat_names[pic_index-8:pic_index]] #visualiza los últimos 8
dog_images = [os.path.join(dog_dir, fname)
	for fname in dog_names[pic_index-8:pic_index]]

#print(len(cat_images),dog_images)

for i, img_path in enumerate(cat_images + dog_images): #ennumerate genera un contador.  se muestran las img
	sp = plt.subplot(4, 4, i+1)
	sp.axis('Off')
	img = mpimg.imread(img_path)
	plt.imshow(img)

#plt.show()

base_dir = 'C:/Programas/datasets/dogs-vs-cats/train/'

# Crear los datasets
# Requiere las carpetas train validation y test en el directorio dogs-vs-cats
train_datagen = tf.keras.utils.image_dataset_from_directory(base_dir,
                                             image_size=(200, 200),
                                             subset='training', #Subset of the data to return. One of "training", "validation" or "both"
                                             seed=1, #Optional random seed for shuffling and transformations.
                                             validation_split=0.1, #entre 0 y 1.  porcentaje para valicacion
                                             batch_size=32) #Size of the batches of data. Default: 32. If None, the data will not be batched (the dataset will yield individual samples).

test_datagen = image_dataset_from_directory(base_dir,
                                                  image_size=(200,200),
                                                  subset='validation',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= 32)

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
#filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
   #ectified linear unit activation function
#input matriz a procesar
#
    layers.MaxPooling2D(2, 2), #Max pooling operation for 2D spatial data.
    #Pooling is a technique used in the CNN model for down-sampling the feature coming from
    # the previous layer and produce the new summarised feature maps.
    # In computer vision reduces the spatial dimensions of an image while retaining important features
    #Max Pooling: Max Pooling selects the maximum value from each set of overlapping
    # filters and passes this maximum value to the next layer.
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    #layers.Conv2D(64, (3, 3), activation='relu'),
    #layers.MaxPooling2D(2, 2),
    #layers.Conv2D(64, (3, 3), activation='relu'),
    #layers.MaxPooling2D(2, 2),
    layers.Flatten(), #transforma ,matriz a vector



    #Dense layer: It's a fully connected layer, it connects every neuron from the previous layer to every neuron in the current layer.

    layers.Dense(512, activation='relu'), #512 dimension espacio de salida. Densde es una red completamente conectada
    layers.BatchNormalization(), #Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1), #tasa de neuronas desactivadas
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()
'''xxxxx
keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)
'''

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)



model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
history = model.fit(train_datagen,
          epochs=1,
          validation_data=test_datagen)


#model.save('C:/Programas/datasets/dogs-vs-cats/modeloentrenado1.keras')
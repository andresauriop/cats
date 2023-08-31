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


#from keras.preprocessing import image
import keras.utils as image

model = keras.models.load_model('C:/Programas/datasets/dogs-vs-cats/modeloentrenado.keras')

# Input image
#test_image = image.load_img('C:/Programas/datasets/dogs-vs-cats/validation/cats/cat.23.jpg', target_size=(200, 200))
test_image = image.load_img('C:/Programas/datasets/dogs-vs-cats/validation/10.jpg', target_size=(200, 200))

# For show image
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Result array
result = model.predict(test_image)
print(result)
# Mapping result array with the main name list
i = 0
if (result >= 0.5):
    print("Dog")
else:
    print("Cat")
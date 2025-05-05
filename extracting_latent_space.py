from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, Dense
import argparse
import copy
import json
import os.path
from enum import Enum
from typing import Dict, Tuple, Union, Optional, Any
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from typing import List, Iterable, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from numpy import ndarray
from tensorflow import keras, Tensor
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Functional
from keras_models import generate_ncp_model
import pandas as pd

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DEFAULT_NCP_SEED = 22222

batch_size = None
seq_len = 64
augmentation_params = None
single_step = True
no_norm_layer = False
mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)

# pretrained model weights
# mymodel.load_weights('model-ncp-val.hdf5')

# custom model weights
root = "./retrain_150traj_wscheduler0.85_seed22222_lr0.001_trainloss0.00035_valloss0.00019_coreset900.h5"
mymodel.load_weights(root)

conv_layers = [layer for layer in mymodel.layers if isinstance(layer, Conv2D)]

# Get the last Conv2D layer
last_conv = conv_layers[-1]

# Get the next two layers after the last Conv2D
layer_dict = {layer.name: layer for layer in mymodel.layers}
next_layer_1 = None
next_layer_2 = None

found = False
for i, layer in enumerate(mymodel.layers):
    if layer.name == last_conv.name:
        next_layer_1 = mymodel.layers[i + 1]
        next_layer_2 = mymodel.layers[i + 2]
        break

# Sanity check
assert isinstance(next_layer_1, Flatten), f"Expected Flatten, got {type(next_layer_1)}"
assert isinstance(next_layer_2, Dense), f"Expected Dense, got {type(next_layer_2)}"

# Create new visualization model
vis_model = Model(
    inputs=mymodel.inputs[0],
    outputs=[last_conv.output, next_layer_1.output, next_layer_2.output]
)

print(vis_model.summary())

base_dir = "dataset"
IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMAGE_SHAPE_CV)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.convert_to_tensor(img_array)
    return img_array


latent_space_data = []

for dir_index in range(len(os.listdir(base_dir))):
    print("Processing directory:", dir_index + 1)
    folder_path = os.path.join(base_dir, str(dir_index + 1))

    files_length = len([file for file in os.listdir(folder_path) if '.png' in file])

    for frame_index in range(files_length - 1):
        image_path = f"{folder_path}/Image{frame_index + 1}.png"
        img = load_image(image_path)
        print(image_path)

        output = vis_model.predict([img])
        latent_vector = output[2]
        latent_space_data.append(latent_vector.squeeze())
       

df = pd.DataFrame(latent_space_data)

# Save to CSV
df.to_csv("latent_space_output.csv", index=False)

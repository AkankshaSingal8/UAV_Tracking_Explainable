import cv2
import time
from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow.python.keras.models import Functional
from tensorflow import keras
import numpy as np
from matplotlib.image import imread
from keras_models import generate_ncp_model
import os
import csv
from PIL import Image
import pandas as pd

def generate_hidden_list(model: Functional, return_numpy: bool = True):
    """
    Generates a list of tensors that are used as the hidden state for the argument model when it is used in single-step
    mode. The batch dimension (0th dimension) is assumed to be 1 and any other dimensions (seq len dimensions) are
    assumed to be 0

    :param return_numpy: Whether to return output as numpy array. If false, returns as keras tensor
    :param model: Single step functional model to infer hidden states for
    :return: list of hidden states with 0 as value
    """
    constructor = np.zeros if return_numpy else tf.zeros
    hiddens = []
    print("Length of model input shape: ", len(model.input_shape))
    if len(model.input_shape)==1:
        lool = model.input_shape[0][1:]
    else:
        print("model input shape: ", model.input_shape)
        lool = model.input_shape[1:]
    print("lool: ", lool)
    for input_shape in lool:  # ignore 1st output, as is this control output
        hidden = []
        for i, shape in enumerate(input_shape):
            if shape is None:
                if i == 0:  # batch dim
                    hidden.append(1)
                    continue
                elif i == 1:  # seq len dim
                    hidden.append(0)
                    continue
                else:
                    print("Unable to infer hidden state shape. Leaving as none")
            hidden.append(shape)
        hiddens.append(constructor(hidden))
    return hiddens

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])


tf.config.set_visible_devices([], 'GPU')

root = './retrain_150traj_wscheduler0.85_seed22222_lr0.001_trainloss0.00035_valloss0.00019_coreset900.h5'

DEFAULT_NCP_SEED = 22222
batch_size = None
seq_len = 64
augmentation_params = None
no_norm_layer = False
single_step = True
model = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)
model.load_weights(root)

base_dir = "dataset"

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMAGE_SHAPE_CV)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.convert_to_tensor(img_array)
    return img_array


prediction_data = []

# Iterate over all directories in base_dir
for dir_index in range(len(os.listdir(base_dir))):
    print("Processing directory:", dir_index + 1)
    folder_path = os.path.join(base_dir, str(dir_index + 1))

    hiddens = generate_hidden_list(model=model, return_numpy=True)
    files_length = len([file for file in os.listdir(folder_path) if '.png' in file])

    for frame_index in range(files_length - 1):
        image_path = f"{folder_path}/Image{frame_index + 1}.png"
        img = load_image(image_path)
        print(image_path)

        output = model.predict([img, *hiddens])
        hiddens = output[1:]
        preds = output[0][0]
        vx, vy, vz, omega_z = preds[0], preds[1], preds[2], preds[3]

        prediction_data.append([vx, vy, vz, omega_z])

df = pd.DataFrame(prediction_data, columns=['vx', 'vy', 'vz', 'omega_z'])

# Save to CSV
df.to_csv("prediction_output.csv", index=False)

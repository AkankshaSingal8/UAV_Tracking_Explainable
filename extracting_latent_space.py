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
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.models import Functional
from keras_models import generate_ncp_model

def generate_latent_model(root):
    IMAGE_SHAPE = (144, 256, 3)
    IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

    DEFAULT_NCP_SEED = 22222
    batch_size = None
    seq_len = 64
    augmentation_params = None
    single_step = True
    no_norm_layer = False

    mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)

    # Load pretrained weights
    mymodel.load_weights(root)

    # Find the output of TimeDistributed(Dense, 128 units)
    dense_layer_output = None
    for layer in mymodel.layers:
        if isinstance(layer, TimeDistributed) and isinstance(layer.layer, Dense) and layer.layer.units == 128:
            dense_layer_output = layer.output
            break

    if dense_layer_output is None:
        raise ValueError("Could not find TimeDistributed(Dense, 128 units) layer.")

    # Create model to extract latent representations
    latent_model = Model(inputs=mymodel.input, outputs=dense_layer_output)
    print(latent_model.summary())
    return latent_model

root = "./retrain_150traj_wscheduler0.85_seed22222_lr0.001_trainloss0.00035_valloss0.00019_coreset900.h5"
generate_latent_model(root)
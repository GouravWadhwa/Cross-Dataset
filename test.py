import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import os
import cv2
import datetime
import random

from model import Model
from dataloader import TestDataset

def test (test_dataset) :
    test_accuracy = 0

    for n, data in tqdm (enumerate (test_dataset)) :
        (x, y) = data
        
        prediction = model (x)
        prediction = tf.math.argmax (prediction, axis=1)
        y = tf.math.argmax (y, axis=1)

        if prediction == y :
            test_accuracy += 1

    print ("Testing Accuracy =", test_accuracy / (n+1))

test_dataset = TestDataset ()

tf.config.experimental_run_functions_eagerly(True)

model = Model ()
model = model.build_model ()

checkpoint_dir = 'Training/training_checkpoints'
if not os.path.isdir (checkpoint_dir) :
    os.mkdir (checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    model=model
)

checkpoint.restore (tf.train.latest_checkpoint (checkpoint_dir))

test (test_dataset)

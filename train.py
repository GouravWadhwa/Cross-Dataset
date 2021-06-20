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
from dataloader import TrainDataset, ValDataset

def calculate_loss (is_labelled, y, predictions) :
    ce_loss = tf.keras.losses.CategoricalCrossentropy () (y, predictions) * is_labelled                                                       # Calculating Cross entropy loss
    ce_loss = tf.math.reduce_sum (ce_loss) / (tf.math.reduce_sum (is_labelled) + 1e-10)                                                       # Normalizing Cross entropy loss

    maximum_predictions = tf.math.reduce_max (predictions, axis=1)                                                                            # Getting the maximum prediction
    cond_max_preds = tf.cast (maximum_predictions > 0.95, tf.float32)                                                                         # Cutoff of 0.95 on maximum predictions
    unlabelled_y = tf.one_hot (tf.math.argmax (predictions, axis=1), depth=CLASSES)                                                           # Getting the index of maximum prediction
    unlabelled_loss = tf.keras.losses.CategoricalCrossentropy () (predictions, unlabelled_y) * (1 - is_labelled) * maximum_predictions        # calculating the cross entropy loss  for unlabelled datasets
    unlabelled_loss = unlabelled_loss * cond_max_preds                                                                                        # Applying the condition
    unlabelled_loss = tf.math.reduce_sum (unlabelled_loss) / (tf.math.reduce_sum ((1-is_labelled) * cond_max_preds) + 1e-10)                  # Normalizing unlabelled loss

    entropy_loss = - predictions * tf.math.log (predictions + 1e-10)                                                                          # Calculating entropy loss
    entropy_loss = tf.math.reduce_sum (entropy_loss) / BATCH_SIZE                                                                             # normalizing entropy loss

    total_loss = ALPHA * ce_loss + BETA * unlabelled_loss + GAMMA * entropy_loss                                                              # Calculating Total Loss

    return ce_loss, unlabelled_loss, entropy_loss, total_loss

@tf.function
def train_step (is_labelled, x, y) :
    with tf.GradientTape () as tape :
        predictions = model (x)
        ce_loss, ce_unlabelled_loss, entropy_loss, total_loss = calculate_loss (is_labelled, y, predictions)

    gradients = tape.gradient (total_loss, model.trainable_variables)
    optimizer.apply_gradients (zip (gradients, model.trainable_variables))

    return ce_loss, ce_unlabelled_loss, entropy_loss, total_loss, predictions

def validate (val_dataset) :
    val_accuracy = 0

    for data in val_dataset :
        (x, y) = data
        
        prediction = model (x)
        prediction = tf.math.argmax (prediction, axis=1)
        y = tf.math.argmax (y, axis=1)

        if prediction == y :
            val_accuracy += 1

    print ("Validation Accuracy =", val_accuracy)
        

def fit (train_dataset, val_dataset, epochs) :
    for epoch in range (epochs) :
        train_accuracy = 0

        avg_total_loss = 0
        avg_ce_loss = 0
        avg_ce_unlabelled_loss = 0
        avg_entropy_loss = 0
    
        for n, data in tqdm (enumerate (train_dataset)) :
            is_labelled, x, y = data

            ce_loss, ce_unlabelled_loss, entropy_loss, total_loss, predictions = train_step (is_labelled, x, y)

            predictions = tf.math.argmax (predictions, axis=1)
            y = tf.math.argmax (y, axis=1)
            
            # print ()
            # print (is_labelled.numpy ().astype (int))
            # print (predictions.numpy ())
            # print (y.numpy ())

            for i in range (tf.shape (y)[0]) :
                if predictions[i] == y[i] :
                    train_accuracy += 1
            
            # print (ce_loss.numpy ())
            # print (ce_unlabelled_loss.numpy ())
            # print (entropy_loss.numpy ())
            # print (total_loss.numpy ())

            plot_ce_loss.append (total_loss.numpy ())
            plot_unlabelled_loss.append (ce_unlabelled_loss.numpy ())
            plot_entropy_loss.append (entropy_loss.numpy ())
            plot_total_loss.append (total_loss.numpy ())

            avg_total_loss += total_loss
            avg_ce_loss += ce_loss
            avg_ce_unlabelled_loss += ce_unlabelled_loss
            avg_entropy_loss += entropy_loss

        avg_total_loss /= (n+1)
        avg_ce_loss /= (n+1)
        avg_ce_unlabelled_loss /= (n+1)
        avg_entropy_loss /= (n+1)

        train_accuracy /= ((n+1)*BATCH_SIZE)

        print ("Total loss =", avg_total_loss)
        print ("Cross Entropy loss =", avg_ce_loss)
        print ("Unlabelled loss =", avg_ce_unlabelled_loss)
        print ("Entropy loss =", avg_entropy_loss)
        print ("Training Accuracy =", train_accuracy)

        validate (val_dataset)
        checkpoint.save (file_prefix=checkpoint_prefix)

RESTORE_CHECKPOINT = True
EPOCHS = 100
BATCH_SIZE = 10
CLASSES = 7

ALPHA = 0.90
BETA = 1.0
GAMMA = 0.1

plot_ce_loss = []
plot_unlabelled_loss = []
plot_entropy_loss = []
plot_total_loss = []

train_dataset = TrainDataset (batch_size=BATCH_SIZE)
val_dataset = ValDataset ()

tf.config.experimental_run_functions_eagerly(True)

model = Model ()
model = model.build_model ()

optimizer = tf.keras.optimizers.Adam (1e-4)

if not os.path.isdir ("Training") :
    os.mkdir ("Training")

checkpoint_dir = 'Training/training_checkpoints'
if not os.path.isdir (checkpoint_dir) :
    os.mkdir (checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    model=model
)

if RESTORE_CHECKPOINT :
    checkpoint.restore (tf.train.latest_checkpoint (checkpoint_dir))

fit (train_dataset, val_dataset, EPOCHS)

plt.plot (plot_ce_loss)
plt.title ("Cross Entropy Loss")
plt.savefig ("Training/Cross_entropy_loss.png")
plt.clf ()

plt.plot (plot_unlabelled_loss)
plt.title ("Unlabelled Loss")
plt.savefig ("Training/Unlabelled_loss.png")
plt.clf ()

plt.plot (plot_entropy_loss)
plt.title ("Entropy Loss")
plt.savefig ("Training/entropy_loss.png")
plt.clf ()

plt.plot (plot_total_loss)
plt.title ("Total Loss")
plt.savefig ("Training/total_loss.png")
plt.clf ()
# this is the classification problem using 
# a global network to guess that shape of the underlying figure. 

# I use the regular keras layuers for simplicity
import tensorflow as tf 
if len(tf.config.list_physical_devices('GPU')) > 0 
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json

from data_generation import generate_data
from gen_coordinates import gen_coord_2d
from interaction_list import comput_inter_list_2d
from layers import PyramidLayer
from layers import DenseLayer
from train import train_step

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

nameScript = sys.argv[0].split('/')[-1]

# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
print("=================================================")
print("Executing " + nameScript + " following " + nameJson, flush = True)
print("=================================================")


# opening Json file # TODO:write a function to manipulate all this
jsonFile = open(nameJson) 
data = json.load(jsonFile)   

# loading the input data from the json file
n_samples = data["nSamples"]              # number of samples 
n_point_samples = data["nPointSamples"]  
L = data["lengthCell"]               # lenght of each cell
r_min = data["minR"]                    # min radious for data
r_max = data["maxR"]                    # max radios for data
radious = data["radious"]
max_num_neighs = data["maxNumNeighbors"]


filterNet = data["filterNet"]
fittingNet = data["fittingNet"]
seed = data["seed"]

filter_channel = filterNet[-1]


# optimization hyper-parame
batch_size = data["batchSize"]
epochsPerStair = data["epochsPerStair"]
learningRate = data["learningRate"]
decayRate = data["decayRate"]
n_epochs = data["numberEpoch"]

# location of the data and model folders
dataFolder = data["dataFolder"]
loadFile = data["loadFile"]

print("We are using the random seed %d"%(seed))
tf.random.set_seed(seed)

dataFile = dataFolder + "data_geom_classification_"+ \
                        "_r_min_%.4f"%(r_min) + \
                        "_r_max_%.4f"%(r_max) + \
                        "_L_%.4f"%(L) + \
                        "_n_samples_" + str(n_samples) + \
                        "_n_samples_per_point_" + str(n_point_samples) + ".h5"

checkFolder  = "checkpoints/"
checkFile = checkFolder + "checkpoint_" + nameScript + \
                          "_json_file_" + nameJson + \
                          "_data_geom_classification_"+ \
                          "_r_min_%.4f"%(r_min) + \
                          "_r_max_%.4f"%(r_max) + \
                          "_L_%.4f"%(L) + \
                          "_n_samples_" + str(n_samples) + \
                          "_n_samples_per_point_" + str(n_point_samples)

print("Using data in %s"%(dataFile))

# if the file doesn't exist we create it
if not path.exists(dataFile):
  # TODO: encapsulate all this in a function
  print("Data file does not exist, we create a new one")

  labels, \
  points = generate_data(n_samples, n_point_samples, L, r_min, r_max)
  
  hf = h5py.File(dataFile, 'w') 
  
  hf.create_dataset('labels', data=labels)   
  hf.create_dataset('points', data=points) 
  
  hf.close()

# extracting the data
hf = h5py.File(dataFile, 'r')

labels_array = np.array(hf['labels'][:], dtype=np.int32)
points_array = np.array(hf['points'][:], dtype=np.float32)


# Defining the Keras model 
model = keras.Sequential()
model.add(keras.Input(shape=(n_point_samples, 2)))  # 250x250 RGB images
model.add(layers.Flatten())
model.add(layers.Dense(filter_channel, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(filter_channel, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(filter_channel, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(filter_channel, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(filter_channel, activation="relu"))
model.add(layers.Dense(2))


# Printing Keras model

print("Training cycles in number of epochs")
print(n_epochs)
print("Training batch sizes for each cycle")
print(batch_size)

### optimization parameters ##
loss_cross = tf.keras.losses.CategoricalCrossentropy(from_logits = True, 
                          reduction=tf.keras.losses.Reduction.SUM)

initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
             initialLearningRate,
             decay_steps=(n_samples//batch_size)*epochsPerStair,
             decay_rate=decayRate,
             staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)

loss_metric = tf.keras.metrics.Mean()

model.compile(optimizer=optimizer, loss=loss_cross, metrics=['accuracy', 'mse'])

model.fit(points_array, labels_array, epochs=n_epochs, validation_split=0.1)



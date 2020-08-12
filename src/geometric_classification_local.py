# typical imports
# we have a simplified deep MD using only the radial information
# and the inverse of the radial information locally
# for the long range we only use
# in this case we initialize the multipliers and we let the method evolve on its own
import tensorflow as tf
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
                        "_L_%.4f"%(r_max) + \
                        "_n_samples_" + str(n_samples) + \
                        "_n_samples_per_point_" + str(n_point_samples) + ".h5"

checkFolder  = "checkpoints/"
checkFile = checkFolder + "checkpoint_" + nameScript + \
                          "_json_file_" + nameJson + \
                          "_data_geom_classification_"+ \
                          "_r_min_%.4f"%(r_min) + \
                          "_r_max_%.4f"%(r_max) + \
                          "_L_%.4f"%(r_max) + \
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

labels_array = hf['labels'][:]
points_array = np.array(hf['points'][:], dtype=np.float32)

labels_array_tf = tf.Variable(labels_array)
points_array_tf = tf.Variable(points_array)


# positions of the 
Rinput = tf.Variable(points_array, name="input", dtype = tf.float32)

# only using the first 100 points 
Rin = Rinput[:10,:,:]

#compute the statistics of the inputs in order to rescale 
#the descriptor computation 
Rinnumpy = Rin.numpy()

Idx = comput_inter_list_2d(Rinnumpy, L,  radious, max_num_neighs)
# dimension are (Nsamples, Npoints and max_num_neighs)
neigh_list = tf.Variable(Idx)

gen_coordinates = gen_coord_2d(Rin, neigh_list, L)


filter = tf.cast(tf.reduce_sum(tf.abs(gen_coordinates), axis = -1)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = 0).numpy()
numTotal = gen_coordinates.shape[0]  


av = tf.reduce_sum(gen_coordinates, 
                    axis = 0, 
                    keepdims =True).numpy()[0]/numNonZero

std = np.sqrt((tf.reduce_sum(tf.square(gen_coordinates - av), 
                             axis = 0, 
                             keepdims=True).numpy()[0] 
                - av**2*(numTotal-numNonZero)) /numNonZero)

print("mean of the inputs are %.8f and %.8f"%(av[0], av[1]))
print("std of the inputs are %.8f and %.8f"%(std[0], std[1]))


class DeepMDClassification(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""
  def __init__(self,
               n_points_sample,
               L, 
               max_num_neighs = 4,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               name='deepMDsimpleEnergy',
               **kwargs):

    super(DeepMDClassification, self).__init__(name=name, **kwargs)

    self.L = L
    # this should be done on the fly, for now we will keep it here
    self.n_points_sample = n_points_sample
    # maximum number of neighbors
    self.max_num_neighs = max_num_neighs
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]
    # we may need to use the tanh here
    self.layerPyramid   = PyramidLayer(descripDim, 
                                       actfn = tf.nn.relu,
                                       initializer = tf.initializers.GlorotUniform())
    self.layerPyramidInv  = PyramidLayer(descripDim, 
                                       actfn = tf.nn.relu,
                                       initializer = tf.initializers.GlorotUniform())
    
    # we may need to use the relu especially here
    self.fittingNetwork = PyramidLayer(fittingDim, 
                                       actfn = tf.nn.relu)
    self.linfitNet      = DenseLayer(2)    

  #@tf.function
  def call(self, inputs, neigh_list):
      
    # (Nsamples, n_points_sample)
    # in this case we are only considering the distances
    gen_coordinates = gen_coord_2d(inputs, 
                                   neigh_list, self.L, 
                                   self.av, self.std) # this need to be fixed
    # (Nsamples*n_points_sample*max_num_neighs, 3)

    # the L1 and L2 functions only depends on the first entry
    L1   = self.layerPyramid(gen_coordinates[:,:1])
    # (Nsamples*n_points_sample*max_num_neighs, descriptorDim)
    L2   = self.layerPyramidInv(gen_coordinates[:,:1])
    # (Nsamples*n_points_sample*max_num_neighs, descriptorDim)
      
    # here we need to assemble the Baby Deep MD descriptor
    gen_Coord = tf.reshape(gen_coordinates, (-1, self.max_num_neighs, 3))
    # (Nsamples*n_points_sample, max_num_neighs, 3)
    L1_reshape = tf.reshape(L1, (-1, self.max_num_neighs, 
                                     self.descriptorDim))
    # (Nsamples*n_points_sample, max_num_neighs, descriptorDim)
    L2_reshape = tf.reshape(L2, (-1, self.max_num_neighs, 
                                     self.descriptorDim))
    # (Nsamples*n_points_sample, max_num_neighs, descriptorDim)

    omega1 = tf.matmul(gen_Coord, L1_reshape, transpose_a = True)
    # (Nsamples*n_points_sample, 3, descriptorDim)
    omega2 =  tf.matmul(gen_Coord, L2_reshape, transpose_a = True)
    # (Nsamples*n_points_sample, 3, descriptorDim)

    D = tf.matmul(omega1, omega2, transpose_a = True)
    # (Nsamples*n_points_sample, descriptorDim, descriptorDim)
    D1 = tf.reshape(D, (-1, model.descriptorDim**2))
    # (Nsamples*n_points_sample, descriptorDim*descriptorDim)
    F2 = self.fittingNetwork(D1)
    F = self.linfitNet(F2)

    Energy = tf.reduce_sum(tf.reshape(F, (-1, self.n_points_sample, 2)),
                           keepdims = False, axis = 1)

    return Energy


# quick run of the model to check that it is correct.
# we use a small set 
av_tf = tf.constant(av, dtype=tf.float32)
std_tf = tf.constant(std, dtype=tf.float32)
L_tf = tf.constant(L, dtype= tf.float32)

## Defining the model
model = DeepMDClassification(n_point_samples, L_tf, max_num_neighs,
                             filterNet, fittingNet, 
                             av_tf, std_tf)

# quick run of the model to check that it is correct.
# we use a small set 
E = model(Rin, neigh_list)
model.summary()

## We use a decent training or a custom one if necessary
if type(n_epochs) is not list:
  n_epochs = [200, 400, 800, 1600]
  #batch_sizeArray = map(lambda x: x*batch_size, [1, 2, 4, 8]) 
  batch_sizeArray = [batch_size*2**i for i in range(0,4)]  
else:  
  assert len(n_epochs) == len(batch_size)
  batch_sizeArray = batch_size

print("Training cycles in number of epochs")
print(n_epochs)
print("Training batch sizes for each cycle")
print(batch_sizeArray)

### optimization parameters ##
loss_cross = tf.keras.losses.CategoricalCrossentropy(from_logits = True, 
                          reduction=tf.keras.losses.Reduction.SUM)

initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
             initialLearningRate,
             decay_steps=(n_samples//batch_sizeArray[0])*epochsPerStair,
             decay_rate=decayRate,
             staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)

loss_metric = tf.keras.metrics.Mean()

for cycle, (epochs, batch_sizeL) in enumerate(zip(n_epochs, batch_sizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batch_sizeL,))

  x_train = (points_array_tf, labels_array_tf)

  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_sizeL)

  # Iterate over epochs.
  for epoch in range(epochs):
    print('============================', flush = True) 
    print('Start of epoch %d' % (epoch,))
  
    loss_metric.reset_states()
  
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):

      Rinnumpy = x_batch_train[0].numpy()
      Idx = comput_inter_list_2d(Rinnumpy, L,  radious, max_num_neighs)
      neigh_list = tf.Variable(Idx)

      # print(neigh_list.shape)
      # print(x_batch_train[0].shape)
      # print(x_batch_train[1].shape)

      loss = train_step(model, optimizer, loss_cross,
                        x_batch_train[0], neigh_list,
                        x_batch_train[1])
      loss_metric(loss)
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))

    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # learning rate using the decay 
    lrStr = str(optimizer._decayed_lr('float32').numpy())
    print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,
                                                          meanLossStr,
                                                          lrStr))

  print("saving the weights")
  model.save_weights(checkFile+"_cycle_"+str(cycle)+".h5")


# print( "values of the trainable decay %f" %(model.NUFFTLayerMultiChannelInit.mu.numpy()))

##### testing ######
pointsTest, \
potentialTest, \
forcesTest  = genDataYukawa(Ncells, Np, mu, 1000, minDelta, L)

forcesTestRscl =  forcesTest- forcesMean
forcesTestRscl = forcesTestRscl/forcesStd

forcePred = model(pointsTest)

err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTestRscl)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
print("Relative Error in the forces is " +str(err.numpy()))


# with tf.GradientTape() as tape:
#   # we watch the inputs 
#   tape.watch(Rinput)
#   # (n_samples, Ncells*Np)
#   # in this case we are only considering the distances
#   genCoordinates = genDistInv(Rinput, model.Ncells, model.Np, 
#                               model.av, model.std)
#   # (n_samples*Ncells*Np*(3*Np - 1), 2)
#   L1   = model.layerPyramid(genCoordinates[:,1:])*genCoordinates[:,1:]
#   # (n_samples*Ncells*Np*(3*Np - 1), descriptorDim)
#   L2   = model.layerPyramidInv(genCoordinates[:,0:1])*genCoordinates[:,0:-1]
#   # (n_samples*Ncells*Np*(3*Np - 1), descriptorDim)
#   # we compute the FMM and the normalize by the number of particules
#   longRangewCoord = model.NUFFTLayerMultiChannelInit(Rinput)
#   # (n_samples, Ncells*Np, 1) # we are only using 4 kernels
#   # we normalize the output of the fmm layer before feeding them to network
#   longRangewCoord2 = tf.reshape(longRangewCoord, (-1, model.fftChannels))
#   # (n_samples*Ncells*Np, 1)
#   L3   = model.layerPyramidLongRange(longRangewCoord2)
#   # (n_samples*Ncells*Np, descriptorDim)

#   # (n_samples*Ncells*Np*(3*Np - 1), descriptorDim)
#   LL = tf.concat([L1, L2], axis = 1)
#   # (n_samples*Ncells*Np*(3*Np - 1), 2*descriptorDim)
#   Dtemp = tf.reshape(LL, (-1, 3*model.Np-1, 
#                           2*model.descriptorDim ))
#   # (n_samples*Ncells*Np, (3*Np - 1), 2*descriptorDim)
#   D = tf.reduce_sum(Dtemp, axis = 1)
#   # (n_samples*Ncells*Np, 2*descriptorDim)

#   DLongRange = tf.concat([D, L3], axis = 1)

#   F2 = model.fittingNetwork(DLongRange)
#   F = model.linfitNet(F2)

#   Energy = tf.reduce_sum(tf.reshape(F, (-1, model.Ncells*model.Np)),
#                           keepdims = True, axis = 1)

# Forces = -tape.gradient(Energy, inputs)

# return Forces



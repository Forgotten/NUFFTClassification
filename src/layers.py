import tensorflow as tf
from tensorflow import keras
import numpy as np 


class DenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, 
                     initializer = tf.initializers.GlorotNormal()):
    super(DenseLayer, self).__init__()
    self.num_outputs = num_outputs
    self.initializer = initializer

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  initializer=self.initializer,
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    self.bias = self.add_weight("bias",
                                initializer=tf.initializers.zeros(),    
                                shape=[self.num_outputs,])


  @tf.function(experimental_relax_shapes=True)
  def call(self, input):
    return tf.matmul(input, self.kernel) + self.bias



# pyramid layer with bias 
class PyramidLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, 
                     actfn = tf.nn.relu,
                     initializer=tf.initializers.GlorotNormal() ):
    super(PyramidLayer, self).__init__()
    self.num_outputs = num_outputs
    self.actfn = actfn
    self.initializer = initializer

  def build(self, input_shape):
    self.kernel = []
    self.bias = []
    self.kernel.append(self.add_weight("kernel",
                       initializer=self.initializer,
                       shape=[int(input_shape[-1]),
                              self.num_outputs[0]]))
    self.bias.append(self.add_weight("bias",
                       initializer=tf.zeros_initializer,
                       shape=[self.num_outputs[0],]))

    for n, (l,k) in enumerate(zip(self.num_outputs[0:-1], \
                                  self.num_outputs[1:])) :

      self.kernel.append(self.add_weight("kernel"+str(n),
                         shape=[l, k]))
      self.bias.append(self.add_weight("bias"+str(n),
                         shape=[k,]))


  @tf.function(experimental_relax_shapes=True)
  def call(self, input):
    # first application
    x = self.actfn(tf.matmul(input, self.kernel[0]) + self.bias[0])
    
    # run the loop
    for k, (ker, b) in enumerate(zip(self.kernel[1:], self.bias[1:])):
      
      # if input equals to output use a shortcut connection
      if self.num_outputs[k] == self.num_outputs[k+1]:
        x += self.actfn(tf.matmul(x, ker) + b)
      # elif 2*self.num_outputs[k] == self.num_outputs[k+1]:
      #   # we try to keep the first features
      #   x = tf.tile(x, [1,2]) + self.actfn(tf.matmul(x, ker) + b)
      
      # if not, just applyt the non-linear layer
      else :
        x = self.actfn(tf.matmul(x, ker) + b)
    return x


class ResNetUnitConv1D(keras.layers.Layer):
  def __init__(self,out_channels):
    super(MLPBlock, self).__init__()

    self.conv_layer1 = layers.Conv1D(filter_channel, 1, strides=1)
    self.batch_norm1 = layers.BatchNormalization()
    self.conv_layer2 = layers.Conv1D(filter_channel, 1, strides=1)
    self.batch_norm2 = layers.BatchNormalization()


  def call(self, inputs):

    x = self.conv_layer1(inputs)
    x = self.batch_norm1(x)
    x = tf.nn.relu(x)
    x = self.conv_layer2(x)
    x = x + inputs
    x = self.batch_norm2(x)

    return tf.nn.relu(x)

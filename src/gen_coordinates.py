import tensorflow as tf
import numpy as np 
from numba import jit 

@tf.function
def gen_coord(Rin, neighList, L, 
              av = tf.constant([0.0], dtype = tf.float32),
              std =  tf.constant([1.0], dtype = tf.float32)):
    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # the input has dimensions (n_samples, Ncells*Np)
    # the ouput has dimensions (n_samples*Ncells*Np*(3*Np-1), 2)
    # we try to allocate an array before hand and use high level
    # tensorflow operations

    # neighList is a (Nsample, Npoints, maxNeigh)

    # add assert with respect to the input shape 
    n_samples = Rin.shape[0]
    # maximum number of neighbors
    max_neighs = neighList.shape[-1]

    mask = neighList > -1

    RinRep  = tf.tile(tf.expand_dims(Rin, -1),[1 ,1,max_neighs])
    RinGather = tf.gather(Rin, neighList, batch_dims = 1, axis = 1)

    # substracting 
    R_Diff = RinGather - RinRep

    # we impose periodicity (To reconsider)
    R_Diff = R_Diff - L*tf.round(R_Diff/L)

    bnorm = (tf.abs(R_Diff) - av[0])/std[0]

    zeroDummy = tf.zeros_like(bnorm)

    bnorm_safe = tf.where(mask, bnorm, zeroDummy)
    
    R_total = tf.reshape(bnorm_safe, (n_samples,-1,max_neighs))

    return R_total
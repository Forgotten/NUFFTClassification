import tensorflow as tf
import numpy as np 


# @tf.function
def gen_coord_2d(r_in, neigh_list, L, 
                 av = tf.constant([0.0, 0.0], dtype = tf.float32),
                 std =  tf.constant([1.0, 1.0], dtype = tf.float32)):
    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)
    # we try to allocate an array before hand and use high level
    # tensorflow operations

    # neigh_list is a (Nsample, Npoints, maxNeigh)

    # add assert with respect to the input shape 
    Nsamples = r_in.shape[0]
    max_num_neighs = neigh_list.shape[-1]

    mask = neigh_list > -1

    # we try per dimension first
    r_in_rep_x = tf.tile(tf.expand_dims(r_in[:,:,0], -1),[1 ,1, max_num_neighs] )
    r_gather_x = tf.gather(r_in[:,:,0], neigh_list, batch_dims = 1, axis = 1)

    r_in_rep_y  = tf.tile(tf.expand_dims(r_in[:,:,1], -1),[1 ,1, max_num_neighs] )
    r_gather_y = tf.gather(r_in[:,:,1], neigh_list, batch_dims = 1, axis = 1)

    # substracting  in X
    r_diff_x = r_gather_x - r_in_rep_x
    r_diff_x = r_diff_x - L*tf.round(r_diff_x/L)

    # substracting in Y 
    r_diff_y = r_gather_y - r_in_rep_y
    r_diff_y = r_diff_y - L*tf.round(r_diff_y/L)

    # computing the norm 
    norm = tf.sqrt(tf.square(r_diff_x) + tf.square(r_diff_y))
    # computing the normalized norm
    # bnorm = (tf.abs(norm) - av[0])/std[0]
    
    binv = tf.math.reciprocal(norm) 

    bx = tf.math.multiply(r_diff_x,binv)
    by = tf.math.multiply(r_diff_y,binv)

    zeroDummy = tf.zeros_like(norm)

    binv_safe = tf.where(mask, (binv- av[0])/std[0], zeroDummy)
    bx_safe = tf.where(mask, bx, zeroDummy)
    by_safe = tf.where(mask, by, zeroDummy)
    
    R_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
                         tf.reshape(bx_safe,    (-1,1)), 
                         tf.reshape(by_safe,    (-1,1)) ], axis = 1)

    return R_total
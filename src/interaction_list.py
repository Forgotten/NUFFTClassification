import numpy as np 
from numba import jit 


@jit(nopython=True)
def comput_inter_list_2d(r_in, L,  radious, max_num_neighs):
  # function to compute the interaction lists 
  # this function in agnostic to the dimension of the data.
  Nsamples, Npoints, dimension = r_in.shape

  # computing the relative coordinates
  DistNumpy =   r_in.reshape(Nsamples,Npoints,1, dimension) \
              - r_in.reshape(Nsamples,1, Npoints,dimension)

  # periodicing the distance
  # working around some quirks of numba with the np.round function
  out = np.zeros_like(DistNumpy)
  # we need to provide an objective array to store the rounded output
  np.round(DistNumpy/L, 0, out)
  
  DistNumpy = DistNumpy - L*out

  # computing the distance
  DistNumpy = np.sqrt(np.sum(np.square(DistNumpy), axis = -1))

  # add the padding and loop over the indices 
  Idx = np.zeros((Nsamples, Npoints, max_num_neighs), dtype=np.int32) -1 
  for ii in range(0,Nsamples):
    for jj in range(0, Npoints):
      ll = 0 
      for kk in range(0, Npoints):
        if jj!= kk and np.abs(DistNumpy[ii,jj,kk]) < radious:
          # checking that we are not going over the max number of
          # neighboors, if so we break the loop
          if ll >= max_num_neighs:
            print("Number of neighboors is larger than the max number allowed")
            break
          Idx[ii,jj,ll] = kk
          ll += 1 

  return Idx

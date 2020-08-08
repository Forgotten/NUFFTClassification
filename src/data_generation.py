# scripts to generate data 

import numpy as np

def create_triange(center, outer_rad): 
	angles = np.random.rand(3,1)*2*np.pi

	coord_x = center[0] + outer_rad*np.cos(angles)
	coord_y = center[1] + outer_rad*np.sin(angles) 
	
	points = np.hcat(coord_x, coord_y)

	return points

def sample_triangle(	)
	# function to sample points in the vertices of the triangle
# scripts to generate data 

import numpy as np


def generate_data(n_samples, n_point_samples, L, r_min, r_max):
  
  # safe guards 
  assert r_min <= r_max
  assert r_max < L/2 

  points_array = np.zeros((n_samples, n_point_samples, 2))
  # we will use a one-hot encoding 
  label_array = np.zeros((n_samples, 2), dtype=int)

  for i in range(n_samples):
    type_rand = np.int(np.random.rand(1)[0] < 0.5)
    
    # we randomly create a center point
    center = np.random.rand(1,2)*L/2 + L/4
    radious = r_min + (r_max - r_min)*np.random.rand(1)[0]

    # we generate a triangle
    if type_rand == 0 :

      # we create the triangle 
      triang_points = create_triangle(center, radious, np.pi/3)
      
      sample_triang = sample_triangle(triang_points, 
                                      n_point_samples)

      points_array[i,:,:] = sample_triang
      label_array[i,:] = np.array([1, 0])

    # we generate a quadrilateral
    if type_rand == 1 :

      # we create the triangle 
      quad_points = create_quadrilateral(center, 
                                         radious, 
                                         np.pi/4)
      
      sample_quad = sample_quadrilateral(quad_points, 
                                        n_point_samples)

      points_array[i,:,:] = sample_quad
      label_array[i,:] = np.array([0, 1])

  return (label_array, points_array)


def sample_side(points_vertices, rel_dist):
  assert 0 <= rel_dist and rel_dist <= 1

  point = rel_dist*points_vertices[0,:] + (1 - rel_dist)*points_vertices[1,:]

  return point


def create_polygon(center, outer_rad, n_sides): 
  # function to create a polygon of n_sides sides
  # within a circle of centered in center and of radious
  # outer rad 

  assert center.shape[0] == 1 and center.shape[1] == 2 
  # choose the angle of the points within the circle
  # we ordered them for cosistency
  angles = np.random.rand(n_sides,1)*2*np.pi

  coord_x = center[0,0] + outer_rad*np.cos(angles)
  coord_y = center[0,1] + outer_rad*np.sin(angles) 
  
  points = np.concatenate([coord_x, coord_y], axis = 1)

  return points

def create_triangle(center, outer_rad, min_angle = 0): 
  # we check that center has the proper dimensions 

  assert center.shape[0] == 1 and center.shape[1] == 2 
  # condition so the triangle can be built
  assert min_angle <= 2*np.pi/3

  angles = np.sort(np.random.rand(3,1)*2*np.pi, axis = 0)

  # we make sure that the circular different is large enough
  while min_circ_dist(angles) < min_angle :
    angles = np.sort(np.random.rand(3,1)*2*np.pi, axis = 0)

  coord_x = center[0,0] + outer_rad*np.cos(angles)
  coord_y = center[0,1] + outer_rad*np.sin(angles) 
  
  points = np.concatenate([coord_x, coord_y], axis = 1)

  return points


def min_circ_dist(angles):
  # utility function to compute maximum angular distance
  dist = angles - np.roll(angles, 1)
  dist_per = np.abs(dist - 2*np.pi*np.round(dist/(2*np.pi)))

  return np.min(dist_per)


def create_quadrilateral(center, outer_rad, min_angle = 0): 
  # we check that center has the proper dimensions 

  assert center.shape[0] == 1 and center.shape[1] == 2 
  # condition so the quadrilateral can be built
  assert min_angle <= 2*np.pi/4 

  angles = np.sort(np.random.rand(4,1)*2*np.pi, axis = 0)

  # we make sure that the circular different is large enough
  while min_circ_dist(angles) < min_angle :
    angles = np.sort(np.random.rand(4,1)*2*np.pi, axis = 0)

  coord_x = center[0,0] + outer_rad*np.cos(angles)
  coord_y = center[0,1] + outer_rad*np.sin(angles) 
  
  points = np.concatenate([coord_x, coord_y], axis = 1)

  return points  

def sample_triangle(points, n_samples):
  # function to sample points in the vertices of the triangle

  assert n_samples >= 3

  rand_pos = 3*np.random.rand(n_samples-3)

  points_out = []

  for pos in rand_pos:
    if pos < 1 :
      points_out.append(sample_side(points[[0, 1], :], pos))
    elif pos < 2 :
      points_out.append(sample_side(points[[1, 2], :], pos-1))
    elif pos < 3 :
      points_out.append(sample_side(points[[0, 2], :], pos-2))

  points_out = np.concatenate([points, np.array(points_out)], axis = 0)

  # adding a random shuffle to the points
  np.random.shuffle(points_out)

  return points_out


def sample_quadrilateral(points, n_samples):
  # function to sample points in the vertices of the triangle

  assert n_samples >= points.shape[0]

  rand_pos = points.shape[0]*np.random.rand(n_samples-points.shape[0])

  points_out = []

  for pos in rand_pos:
    if pos < 1 :
      points_out.append(sample_side(points[[0, 1], :], pos))
    elif pos < 2 :
      points_out.append(sample_side(points[[1, 2], :], pos-1))
    elif pos < 3 :
      points_out.append(sample_side(points[[2, 3], :], pos-2))
    elif pos < 4 :
      points_out.append(sample_side(points[[3, 0], :], pos-3))

  points_out = np.concatenate([points, np.array(points_out)], axis = 0)

  # adding a random shuffle to the points
  np.random.shuffle(points_out)

  return points_out

def main():
  print("testing the generation of data")

  print("testing the sample side")

  ss = np.array([[0.0,0.0], [1.0, 0.0]])
  mid_point = sample_side(ss, 0.5 ) 

  assert mid_point[0] == 0.5 and mid_point[1] == 0.0

  center = np.zeros((1,2))
  triangle_points = create_triangle(center, 1.0)
  dist_points = np.sqrt(np.sum(np.square(triangle_points), axis = 1))
  assert  dist_points.all() == 1.0

  sample_triangle(triangle_points, 6)

  quad_points = create_quadrilateral(center, 1.0)  
  dist_points = np.sqrt(np.sum(np.square(quad_points), axis = 1))
  assert  dist_points.all() == 1.0

  sample_quad = sample_quadrilateral(quad_points, 12)

if __name__ == "__main__":
    main()


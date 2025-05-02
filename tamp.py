import matplotlib.pyplot as pt
import numpy as np
import torch as tr
from scipy.optimize import minimize, NonlinearConstraint
import pybullet as pb
import itertools as it
import random

CUBE_SIDE = 0.01905

def check_collision(env, current_location,total_locations):
    for loc in total_locations:
        dist = np.linalg.norm(np.array(current_location) - np.array(loc))
        if dist < CUBE_SIDE*2:
            return False
    return True    
    

def get_spare_positions(env,total_locations):
    expected = len(env.block_id)
    free_positions = []
    i = 0
    r = [-.16,-.11,-.11,-.11,-.11,-.13]
    thetas = [1.0, 1.2,1.6,2.1,2.5,2.7] 
    while len(free_positions) < expected:
        
        theta = thetas[i]
        pos = (r[i]*np.cos(theta), r[i]*np.sin(theta), -0.015525)
        # if check_collision(env, pos, total_locations) == False:
        free_positions.append(pos)
        i += 1
    return free_positions

    

def sort_by_third_element(vector_list):
  """Sorts a list of 3D vectors based on their third element.

  Args:
    vector_list: A list of 3D vectors (lists or tuples).

  Returns:
    A new list with the vectors sorted by their third element.
  """
  return sorted(vector_list, key=lambda vector: vector[2])


def tamp(env, goal_poses):
    total_locations = []
    for _,values in goal_poses.items():
        total_locations.append(values[0])
    
    for id in env.block_id:
        loc, _ = pb.getBasePositionAndOrientation(env.block_id[id])
        total_locations.append(loc)
        
    init_positions = []
    for id in env.block_id:
        loc, _ = pb.getBasePositionAndOrientation(env.block_id[id])
        init_positions.append(loc)    
    
    spare_locations = get_spare_positions(env,total_locations)
    return spare_locations
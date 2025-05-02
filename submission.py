"""
Example controller to illustrate API methods and conventions
"""
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as pt
from simulation import SimulationEnvironment
import evaluation as ev
import inverse_kinematics as ik
import pybullet as pb
import torch as tr
from tamp import tamp
from forward_kinematics import get_frames
CUBE_SIDE = 0.01905

def sort_by_third_element(vector_list):
  """Sorts a list of 3D vectors based on their third element.

  Args:
    vector_list: A list of 3D vectors (lists or tuples).

  Returns:
    A new list with the vectors sorted by their third element.
  """
  return sorted(vector_list, key=lambda vector: vector[2])



class Controller:
    def __init__(self):
        # no optimized model data in this example
        pass
    
    def run(self, env, goal_poses,grad_list):
        # run the controller in the environment to achieve the goal
        # this example ignores goal_poses
        # just follows a hand-coded trajectory for sake of example
        # get starting angles for trajectory
        
        trajectory = []
        current = 0
        spare_positions = tamp(env,goal_poses)
        goal_poses = list(goal_poses.values())
        
        block_locations,block_orientations = [],[]
        goal_locations,goal_orientations = [],[]

        for i,id in enumerate(env.block_id):
            if id == "b0": continue
            
            loc, ori = pb.getBasePositionAndOrientation(env.block_id[id])
            block_locations.append(loc+(i,))
            block_orientations.append(ori)
            goal_locations.append(goal_poses[i-1][0] + (i-1,))
            goal_orientations.append(goal_poses[i-1][1])

        block_locations = sort_by_third_element(block_locations)
        goal_locations = sort_by_third_element(goal_locations)
        block_locations.reverse()
        
        current = 0
        for block in block_locations:
            id = block[3]
            block = block[:3]
            block = list(block)
            loc = tr.tensor(block)
            block[2] -= 1.8*0.0327993216120967
            targets = [block,block]
            zeroth_step = {}
            
            zeroth_step = dict(ik.get_angles(env,targets))
            zeroth_step['m2'] -= 30.
            trajectory.append(zeroth_step)
            
            first_step = dict(zeroth_step)
            first_step['m1'] += 8.
            first_step['m6'] += 40.
            first_step['m2'] += 30.
            trajectory.append(first_step)
          
            second_step = dict(first_step)
            second_step['m6'] -= 25
            trajectory.append(second_step)
            
            third_step = dict(second_step)
            third_step['m2'] -= 30.
            trajectory.append(third_step)
            
            block = spare_positions[current]
            loc = tr.tensor(block)
            targets = [loc,loc]
            forth_step = dict(ik.get_angles(env,targets))
            forth_step['m6'] = third_step['m6']
            trajectory.append(forth_step)
            
            fifth_step = dict(forth_step)
            fifth_step['m6'] += 30.
            trajectory.append(fifth_step)
            
            sixth_step = dict(fifth_step)
            sixth_step['m2'] -= 30.
            trajectory.append(sixth_step)
            
            current += 1
        duration = 10. # duration of each waypoint transition (in seconds)
        for waypoint in trajectory:
            env.goto_position(waypoint, duration)
            
        trajectory = []
        
        for goal_location in goal_locations:
            id = 'b' + str(goal_location[3]+1)
            
            loc, ori = env.get_block_pose(id)
            goal_location = goal_location[:3]
            
            loc = tr.tensor(loc)
            loc[2] -= 1.8*0.0327993216120967
            targets = [loc,loc]
            zeroth_step = dict(ik.get_angles(env,targets))
            zeroth_step['m2'] -= 40.
            trajectory.append(zeroth_step)
            
            first_step = dict(zeroth_step)
            first_step['m1'] += 10.
            first_step['m6'] += 40.
            first_step['m2'] += 40.
            trajectory.append(first_step)
          
            second_step = dict(first_step)
            second_step['m6'] -= 25
            trajectory.append(second_step)
            
            
            third_step = dict(second_step)
            third_step['m2'] -= 30.
            trajectory.append(third_step)
            
            block = goal_location
            
            loc = tr.tensor(block)
            loc[2] -= 1.9*0.0327993216120967
            targets = [loc,loc]
            
            forth_step = dict(ik.get_angles(env,targets))
            forth_step['m6'] = third_step['m6']
            forth_step['m2'] -= 30.
            forth_step['m1'] += 5
            trajectory.append(forth_step)
            
            mid_step = dict(forth_step)
            mid_step['m2'] += 30.
            trajectory.append(mid_step)
            
            
            fifth_step = dict(mid_step)
            fifth_step['m6'] += 10.
            trajectory.append(fifth_step)
            
            sixth_step = dict(fifth_step)
            sixth_step['m2'] -= 30.
            trajectory.append(sixth_step)
            
        # runs the trajectory
        duration = 60. # duration of each waypoint transition (in seconds)
        for waypoint in trajectory:
            env.goto_position(waypoint, duration)
            
        fin_loc = []
        for val in env.block_id:
            loc,_ = env.get_block_pose(id)
            fin_loc.append(loc)

if __name__ == "__main__":
    accuracy_list,mean_loc_err,max_loc_err,mean_rot_err,max_rot_err = [],[],[],[],[]
    
    controller = Controller()
    
    grad_list = []
    
    max_iter = 1
    
    for i in range(max_iter):
        grad_list = []
        # sample a validation trial
        env, goal_poses = ev.sample_trial(num_blocks=5, num_swaps=2, show=True)
            
                
        # run the controller on the trial
        # copies goal_poses in case your controller modifies it (but you shouldn't)
        controller.run(env, dict(goal_poses),grad_list)

        # evaluate success
        accuracy, loc_errors, rot_errors = ev.evaluate(env, goal_poses)
        input("Enter")
        env.close()
        accuracy_list.append(int(100*accuracy))
        mean_loc_err.append(float(f"{np.mean(loc_errors):.3f}"))
        max_loc_err.append(float(f"{np.max(loc_errors):.3f}"))
        
        mean_rot_err.append(float(f"{np.mean(rot_errors):.3f}"))
        max_rot_err.append(float(f"{np.max(rot_errors):.3f}"))
        
        print(f"\n{int(100*accuracy)}% of blocks near correct goal positions")
        print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
        print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")
        
    
    # Ensure integer ticks on x-axis
    pt.xticks(range(1,max_iter+1))
    pt.plot(range(1,max_iter+1), accuracy_list, label='Accuracy in %')
    pt.title('Accuracy')
    pt.xlabel('Iterations')
    pt.ylabel('Accuracy %100')
    
    pt.savefig("accuracy.png")
    
    pt.clf()
    
    
    pt.xticks(range(1,max_iter+1))
    pt.plot(range(1,max_iter+1), mean_loc_err, label='Mean Location Error')
    pt.plot(range(1,max_iter+1), max_loc_err, label='Max Location Error')
    pt.title('Mean and Max Location Error')
    pt.xlabel('Iterations')
    pt.ylabel('Mean and Max Error')
    pt.legend()
    pt.savefig("Location_Error.png")
    pt.clf()
    
    pt.xticks(range(1,max_iter+1))
    pt.plot(range(1,max_iter+1), mean_rot_err, label='Mean Rotation Error')
    pt.plot(range(1,max_iter+1), max_rot_err, label='Max Rotation Error')
    pt.title('Mean and Max Orientation Error')
    pt.xlabel('Iterations')
    pt.ylabel('Mean and Max Error')
    pt.legend()
    pt.savefig("Orientation_Error.png")
    pt.clf()
    
    # print(len(grad_list),"len")
    # pt.xticks([x*10 for x in range(int(len(grad_list)/10))])
    # pt.plot([x for x in range(len(grad_list))], grad_list, label='Gradient')
    # pt.title('Grad')
    # pt.xlabel('Iterations')
    # pt.ylabel('Gradient')
    # pt.legend()
    # pt.savefig("grad.png")
    
    
    
    
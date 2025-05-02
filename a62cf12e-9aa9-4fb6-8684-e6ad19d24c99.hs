"""
Example controller to illustrate API methods and conventions
"""
import numpy as np
import matplotlib.pyplot as pt
from simulation import SimulationEnvironment
import evaluation as ev
import inverse_kinematics as ik
import pybullet as pb
import torch as tr
from tamp import tamp
from forward_kinematics import get_frames
CUBE_SIDE = 0.01905

class ExampleController:
    def __init__(self):
        # no optimized model data in this example
        
        pass

    # def forward_kinematics()
    def run(self, env, goal_poses):
        # run the controller in the environment to achieve the goal
        # this example ignores goal_poses
        # just follows a hand-coded trajectory for sake of example
        # print(goal_poses,"here")
        # get starting angles for trajectory
        # init_angles = env.get_current_angles()
        cord_trajectories = tamp(env, goal_poses)
        trajectory = []
        for id in env.block_id:
            
            loc, _ = pb.getBasePositionAndOrientation(env.block_id[id])
            jointPoses = pb.calculateInverseKinematics(env.robot_id, 5, loc)
            frames = get_frames(env.get_joint_info() ,tr.tensor(list(jointPoses)))
            
            targets = [tr.tensor(loc),tr.tensor(loc)]
            first_step = dict(ik.get_angles(env,targets))
            print(frames[5],frames[7],'pb_answer')
            frames = get_frames(env.get_joint_info() ,tr.tensor(list(first_step.values())))
            print(frames[5],frames[7],'ik_answer')
            # print(frames[0][5],frames[0][7],targets,"here")
            # print(loc,forward_kinema(first_step),"here\n")
            trajectory.append(first_step)
        
        # for traj in cord_trajectories:
        #     targets = [tr.tensor(traj),tr.tensor(traj)]
        #     first_step = dict(ik.get_angles(env,targets))
        #     trajectory.append(first_step)
        #     print(traj,"traj")
        
        # current = 0
        # while current < 5:
        #     r = np.random.rand()*(-.2) - 0.1
        #     theta = np.random.rand()*3.14
        #     pos = (r*np.cos(theta), r*np.sin(theta), -0.5*CUBE_SIDE)
        #     # print(pos,"pos",theta)
            
        #     targets = [tr.tensor(pos),tr.tensor(pos)]
        #     first_step = dict(ik.get_angles(env,targets))
        #     trajectory.append(first_step)
        #     print(pos,"pos")
        #     current += 1
                    
        # for traj in enumerate(cord_trajectories):
            
            # targets = [tr.tensor(traj[0]),tr.tensor([.2, -.1, 0])]
            # first_step = dict(ik.get_angles(env,targets))
            # trajectory.append(first_step)
        
        
        
        
        # # move arm down around block
        # stage_angles = dict(first_step)
        # stage_angles["m2"] = 60. # second motor angle (degrees)
        # stage_angles["m3"] = 30. # second motor angle (degrees)
        # trajectory.append(stage_angles)

        # # # close the gripper (sixth motor)
        # close_angles = dict(stage_angles)
        # close_angles["m6"] = -8.49
        # trajectory.append(close_angles)

        # # # lift arm
        # lift_angles = dict(close_angles)
        # # stage_angles["m2"] = -5.
        # lift_angles["m3"] = -90
        # trajectory.append(lift_angles)

        # # # rotate arm  
        # rotate_angles = dict(lift_angles)
        # rotate_angles["m1"] = -33
        # trajectory.append(rotate_angles)

        # # # lower arm
        # lower_angles = dict(rotate_angles)
        # lower_angles["m3"] = 35
        # trajectory.append(lower_angles)

        # # # release and lift arm
        # release_angles = dict(lower_angles)
        # release_angles["m3"] = 20
        # release_angles["m6"] = 0
        # trajectory.append(release_angles)

        # runs the trajectory
        duration = 60. # duration of each waypoint transition (in seconds)
        for waypoint in trajectory:
            env.goto_position(waypoint, duration)

if __name__ == "__main__":

    # # initialize controller class
    controller = ExampleController()

    

    # sample a validation trial
    env, goal_poses = ev.sample_trial(num_blocks=5, num_swaps=1, show=False)

    # run the controller on the trial
    # copies goal_poses in case your controller modifies it (but you shouldn't)
    controller.run(env, dict(goal_poses))

    # evaluate success
    accuracy, loc_errors, rot_errors = ev.evaluate(env, goal_poses)

    env.close()

    print(f"\n{int(100*accuracy)}% of blocks near correct goal positions")
    print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
    print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")


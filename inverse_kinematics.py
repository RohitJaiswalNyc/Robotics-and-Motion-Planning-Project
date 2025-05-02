# Implement the stubs where requested in the comments below
# No imports allowed other than these, do not change these lines
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as pt
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller
import forward_kinematics as fk


def location_constraint_fun(joint_info, i, target_t, angles_t):
    """
    Constraint function for a joint location to coincide with a target
    Input:
        joint_info: same format as fk.get_frames
        i: the index of the joint to be constrained
        target_t: shape (3,) tensor with the target location for the joint
        angles_t: shape (J,) tensor with the candidate joint angles (no batching)
    Output:
        sq_dist: shape (1,) tensor of the squared distance from the joint location to the target
    The joint location should be determined by fk.  Batches are not supported.
    J is the number of non-fixed joints only (where axes in joint_info is not None)
    """
    
    locations, _ = fk.get_frames(joint_info, angles_t)
    diff = locations[i] - target_t
    
    return (diff @ diff).unsqueeze(0)


def location_constraint_jac(joint_info, i, target_t, angles_t,grad_list):
    """
    Constraint function jacobian for the location constraint
    Input: same format as location_constraint_fun
    Output: jac, a shape (1, J) tensor of the jacobian of the constraint
    """
    locations, orientations = fk.get_frames(joint_info, angles_t)
    jacobian = fk.get_jacobian(joint_info, locations, orientations) 
    diff = locations[i] - target_t  

    grad = (2 * diff) @ jacobian[i] 
    grad = grad.unsqueeze(dim=0)
    return grad



def location_constraint(joint_info, i, target_t,grad_list):
    """
    Wraps the location constraint function and jacobian in an so.NonlinearConstraint object
    """

    # convert to/from numpy
    def fun(angles_n):
        angles_t = tr.tensor(angles_n)
        sq_dist_t = location_constraint_fun(joint_info, i, target_t, angles_t)
        return sq_dist_t.numpy()
    def jac(angles_n):
        angles_t = tr.tensor(angles_n)
        jac_t = location_constraint_jac(joint_info, i, target_t, angles_t,grad_list)
        return jac_t.numpy()
    lo = hi = np.zeros(1) # constraint function should be 0 when satisfied
    
    return so.NonlinearConstraint(fun, lo, hi, jac)

def convert_angles(angles):
    """
    Converts angles from 5 acctuators to 7 joints
    """
    ret = np.zeros(8)
    ret[0] = angles[0]
    ret[1] = angles[1]
    ret[2] = angles[2]
    ret[3] = angles[3]
    ret[4] = angles[4]
    ret[5] = 0
    ret[6] = angles[5]
    ret[7] = 0
    return ret

def angle_norm_obj_and_grad(angles_n):
    """
    so.minimize objective function for the squared angle vector norm
    returns the objective value and its gradient
    """
    return (angles_n @ angles_n), 2*angles_n

def orientation_constraint_fun(joint_info, i, target_ori, angles_t):
    _, orientations = fk.get_frames(joint_info, angles_t)
    diff = orientations[i] - target_ori
    sq_dist = (diff @ diff).unsqueeze(0)
    return sq_dist

def orientation_constraint_jac(joint_info, i, target_ori, angles_t):
    angles_t = angles_t.detach().clone().requires_grad_(True)
    def f(a): return fk.get_frames(joint_info, a)[1][i]
    q = f(angles_t)
    loss = ((q - target_ori) @ (q - target_ori))
    grad = tr.autograd.grad(loss, angles_t)[0]  # Extract tensor from tuple
    grad = grad.view(1, -1)  # Reshape to (1, 6)
    return grad

def orientation_constraint(joint_info, i, target_ori):
    def fun(angles_n):
        angles_t = tr.tensor(angles_n)
        return orientation_constraint_fun(joint_info, i, target_ori, angles_t).numpy()
    def jac(angles_n):
        angles_t = tr.tensor(angles_n)
        return orientation_constraint_jac(joint_info, i, target_ori, angles_t).numpy()
    lo = hi = np.zeros(1)
    return so.NonlinearConstraint(fun, lo, hi, jac)



def get_angles(env, targets,grad_list = []):
    joint_info = env.get_joint_info()
    # Define constraints for joint 5 and 7 (gripper tips)
    constraints = [
        location_constraint(joint_info, 5, tr.tensor(targets[0]),grad_list),
        location_constraint(joint_info, 7, tr.tensor(targets[1]),grad_list),
    ]

    # Optimization using scipy
    soln = so.minimize(
        angle_norm_obj_and_grad,               # Objective function
        x0 = np.zeros(6),                      # Initial guess for 6 actuator joints
        jac = True,                            # Gradient of objective
        bounds = [(-np.pi, np.pi)] * 6,        # Joint limits
        constraints = constraints,             # Location constraints
        options = {'maxiter': 50000},
    )

    if not soln.success:
        raise RuntimeError(f"Optimization failed: {soln.message}")

    angles = env._angle_dict_from(convert_angles(soln.x), convert=True)
    
    return angles

    

# if __name__ == "__main__":

    # target = tr.tensor([.1, .2, .3])
    # angles = tr.full((6,), .2)
    # print(location_constraint_fun(ergo_jr.joint_info, 4, target, angles))
    # print(location_constraint_jac(ergo_jr.joint_info, 4, target, angles))

    # target_5 = tr.tensor([.0, -.07, 0])
    # target_7 = tr.tensor([-.02, -.07, 0])

    # print(location_constraint_fun(ergo_jr.joint_info, 5, target_5, tr.zeros(6)))
    # print(location_constraint_jac(ergo_jr.joint_info, 5, target_5, tr.zeros(6)))

    # soln = so.minimize(
    #     angle_norm_obj_and_grad,
    #     x0 = np.zeros(6),
    #     jac = True,
    #     bounds = [(-np.pi, np.pi)] * 6,
    #     constraints = [
    #         location_constraint(ergo_jr.joint_info, 5, target_5),
    #         location_constraint(ergo_jr.joint_info, 7, target_7),
    #     ],
    #     options={'maxiter': 200},
    # )

    # print(soln.message)

    # angles_t = tr.tensor(soln.x)
    # print(soln.x)
    # locations, orientations = fk.get_frames(ergo_jr.joint_info, angles_t)

    # ax = pt.gcf().add_subplot(projection='3d')
    # ax.plot(*locations.numpy().T, 'ko-')
    # ax.plot(*target_5.numpy(), 'ro')
    # ax.plot(*target_7.numpy(), 'bo')
    # pt.axis("equal")
    # pt.show()



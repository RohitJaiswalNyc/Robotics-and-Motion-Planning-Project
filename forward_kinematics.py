# Implement the stubs where requested in the comments below
# No imports allowed other than these, do not change these lines
import matplotlib.pyplot as pt
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller
import rotations as rt

def get_frames(joint_info, joint_angles):
    
    # batched checks if batched
    batched = (len(joint_angles.shape) == 2)
    if batched == 0: joint_angles = joint_angles.unsqueeze(dim=0)

    # index of all actuators [0,1,2,3,4,6]
    ind = [i for i, axis in enumerate([element[4] for element in joint_info]) if axis != None]

    # initialize locations,orientations with first joint location, identity rotation versor
    locations = [tr.tensor(joint_info[0][2]).expand(joint_angles.shape[0], -1)]
    orientations = [rt.identity_versor(batch_dims=joint_angles.shape[0])]

    for i in range(1, len(joint_info)):
        _,par,t,R,_ = joint_info[i]

        parent_location = locations[par]
        parent_orientation = orientations[par]
        t = tr.tensor(t)
        R = tr.tensor(R).expand(joint_angles.shape[0], -1)
        
        par_idx = ind.index(par)
        parent_axis = tr.tensor(joint_info[par][4])
        parent_axis_rot = rt.versor_from(parent_axis,joint_angles[...,par_idx])
        
        # child_location and orientation wrt base
        child_location = parent_location + rt.rotate(rt.multiply(parent_orientation,parent_axis_rot),t)
        child_orientation = rt.multiply(rt.multiply(parent_orientation,parent_axis_rot),R)
        
        locations.append(child_location)
        orientations.append(child_orientation)

    locations = tr.stack(locations, dim=1)
    orientations = tr.stack(orientations, dim=1)
    
    if batched == 0:
        locations = locations.squeeze(dim=0)
        orientations = orientations.squeeze(dim=0)
    return locations, orientations


def get_jacobian(joint_info, locations, orientations):
    
    batched = len(locations.shape) == 2
    if batched:
        locations = locations.unsqueeze(dim=0)
        orientations = orientations.unsqueeze(dim=0)
    assert(len(locations.shape) == 3)
    
    batch, tot_joints, _ = locations.shape
    total_movable_joints = sum(element[4] != None for element in joint_info)

    jacobian = tr.zeros(batch, tot_joints, 3, total_movable_joints)

    ind = [i for i, axis in enumerate([element[4] for element in joint_info]) if axis != None]
    indexes = {}
    
    for i,x in enumerate(ind): indexes[x] = i

    # tree search to update jacobian wrt joint i and current being its ancestors
    for i in range(tot_joints):
        current = i
        while current != -1:
            if current not in indexes:
                current = joint_info[current][1]
                pass
            assert(current in indexes)
            indx = indexes[current]
            new_ori = rt.rotate(orientations[:, current], tr.tensor(joint_info[current][-1]))
            jacobian[:, i, :, indx] = tr.cross(new_ori, locations[:, i, :] - locations[:, current, :])
            current = joint_info[current][1]
    
    if batched: jacobian = jacobian.squeeze(0)
    return jacobian




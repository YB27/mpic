''' Operations and functions related to SE(2) poses '''
''' Poses are supposed expressed as q = [x, y, theta] '''

import math_utility
import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats
import copy

''' TODO : use correct marginalisation :) '''
def fromPosePDF3DEuler(posePDF3D):
    mean = np.empty((3,))
    mean[0:2] = posePDF3D["pose_mean"][3:5]
    mean[2] = posePDF3D["pose_mean"][0]
    cov = np.empty((3,3))
    cov[0:2,0:2] = posePDF3D["pose_cov"][3:5,3:5]
    cov[2,2] = posePDF3D["pose_cov"][0,0]
    cov[0,2] = posePDF3D["pose_cov"][0,3]
    cov[2,0] = cov[0,2]
    cov[1,2] = posePDF3D["pose_cov"][0,4]
    cov[2,1] = cov[1,2]
    return {"pose_mean": mean, "pose_cov": cov}

def fromPosePDF2DTo3D(posePDF2D, posePDF3D):
    pose3D = copy.deepcopy(posePDF3D)
    pose3D["pose_mean"][0] = posePDF2D["pose_mean"][2]
    pose3D["pose_mean"][3] = posePDF2D["pose_mean"][0]
    pose3D["pose_mean"][4] = posePDF2D["pose_mean"][1]

    pose3D["pose_cov"][0, 0] = posePDF2D["pose_cov"][2, 2]
    pose3D["pose_cov"][3:5, 3:5] = posePDF2D["pose_cov"][0:2, 0:2]
    pose3D["pose_cov"][0, 3] = posePDF2D["pose_cov"][0, 2]
    pose3D["pose_cov"][3, 0] = pose3D["pose_cov"][0, 3]
    pose3D["pose_cov"][0, 4] = posePDF2D["pose_cov"][1, 2]
    pose3D["pose_cov"][4, 0] = pose3D["pose_cov"][0, 4]

    return pose3D

def rotationMatrix(q):
    cos = np.cos(q[2])
    sin = np.sin(q[2])
    return np.array([[cos, -sin],
                     [sin, cos]])
def composePose(q1, q2):
    R2 = rotationMatrix(q2)
    t_comp = R2@q1[0:2] + q2[0:2]
    return np.array([t_comp[0], t_comp[1], q1[2] + q2[2]])

def composePosePoint(q, point):
    R = rotationMatrix(q)
    return R@point + q[0:2]

def composePosePoint_array(q, point_array):
    R = rotationMatrix(q)
    return np.einsum('ij,kj->ki', R,  point_array) + q[0:2]

def computeJacobian_composePose(q1, q2):
    cos_2 = np.cos(q2[2])
    sin_2 = np.sin(q2[2])
    jacobian_q2 = np.array([[1., 0., -(sin_2*q1[0] +  cos_2*q1[1])],
                             [0., 1., cos_2*q1[0] - sin_2*q1[1]],
                             [0., 0., 1.]])
    jacobian_q1 = np.zeros((3, 3))
    jacobian_q1[0:2,0:2] = rotationMatrix(q2)
    jacobian_q1[2,2] = 1.

    return jacobian_q1, jacobian_q2

def composePosePDF(q1, q2):
    jacobian_q1, jacobian_q2 = composeJacobian_composePose(q1, q2)
    return {"pose_mean": composePose(q1["pose_mean"], q2["pose_mean"]), "pose_cov": jacobian_q1@q1["pose_cov"]@jacobian_q1.t +
                                                                                     jacobian_q2@q2["pose_cov"]@jacobian_q2.t}
def jacobian_composePosePoint(q, point):
    cos = np.cos(q[2])
    sin = np.sin(q[2])
    jacobian_q = np.array([[1., 0., -sin*point[0] - cos*point[1]],
                           [0., 1., cos*point[0] - sin*point[1]]])
    jacobian_point = rotationMatrix(q)

    return jacobian_q, jacobian_point

def jacobian_composePosePoint_array(q, point_array):
    cos = np.cos(q[2])
    sin = np.sin(q[2])
    jacobian_q = np.empty((point_array.shape[0], 2, 3))
    jacobian_q[:,0:2,0:2] = np.eye(2)
    jacobian_q[:, 0, 2] = -sin*point_array[:,0] - cos*point_array[:,1]
    jacobian_q[:, 1, 2] = cos*point_array[:,0] - sin*point_array[:,1]

    jacobian_point = rotationMatrix(q)
    return jacobian_q, jacobian_point

def composePosePDFPoint(q, point):
    jacobian_q, jacobian_point = jacobian_composePosePoint(q["pose_mean"], point["mean"])
    return {"mean": composePosePoint(q["pose_mean"], point["mean"]), "cov": jacobian_q@q["pose_cov"]@jacobian_q.T +
                                                                            jacobian_point@point["cov"]@jacobian_point.T}

def composePosePDFPoint_array(q, point_array):
    jacobian_q, jacobian_point = jacobian_composePosePoint_array(q["pose_mean"], point_array["mean"])
    cov = np.einsum('kij,kjl->kil', jacobian_q, np.einsum('ij,klj->kil',q['pose_cov'],jacobian_q,optimize=True),optimize=True) + \
          np.einsum('ij,kjl->kil', jacobian_point,
                    np.einsum('kij,lj->kil', point_array['cov'], jacobian_point,
                              optimize=True), optimize=True)

    return {"mean" : composePosePoint_array(q["pose_mean"], point_array["mean"]), "cov" : cov}
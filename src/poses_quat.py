''' Operations and functions related to SE(3) poses
    Implementation based on :
    "J.-L. Blanco. A tutorial on se (3) transformation parameterizations and
     on-manifold optimization. University of Malaga, Tech. Rep, 3, 2010."
'''

import math_utility
import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats
import poses_euler

''' Convert quaternion format from [qr qx qy qz] (MRPT) to [qx qy qz qr] (numpy) '''
def fromqrxyzToqxyzr(q):
    return np.array([q[1], q[2], q[3], q[0]])

def fromqrxyzToqxyzr_array(q):
    res = np.empty((q.shape[0],4))
    res[:, 0] = q[:, 1]
    res[:, 1] = q[:, 2]
    res[:, 2] = q[:, 3]
    res[:, 3] = q[:, 0]
    return res  # np.array([q[3], q[0], q[1], q[2]])

''' Convert quaternion format from [qx qy qz qr] (numpy) to [qr qx qy qz] (MRPT) '''
def fromqxyzrToqrxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])

''' Convert quaternion format from [qx qy qz qr] (numpy) to [qr qx qy qz] (MRPT) '''
def fromqxyzrToqrxyz_array(q):
    res = np.empty((q.shape[0],4))
    res[:,0] = q[:,3]
    res[:,1] = q[:,0]
    res[:,2] = q[:,1]
    res[:,3] = q[:,2]
    return res #np.array([q[3], q[0], q[1], q[2]])

''' q1 + q2, Quaternion '''
def composePoseQuat(q1,q2):
    ''' TODO : implement without using rotation matrix '''
  #  quat_arr_1 = q1[0:4]
  #  quat_arr_2 = q2[0:4]
    quat_arr_1 = fromqrxyzToqxyzr(q1)# np.array([q1[1], q1[2], q1[3], q1[0]])
    quat_arr_2 = fromqrxyzToqxyzr(q2)#np.array([q2[1], q2[2], q2[3], q2[0]])

    rot1 = scipy.spatial.transform.Rotation.from_quat(quat_arr_1)
    rot2 = scipy.spatial.transform.Rotation.from_quat(quat_arr_2)
    t1 = q1[4:7]
    t2 = q2[4:7]

    rot_12 = scipy.spatial.transform.Rotation.from_dcm(np.matmul(rot1.as_dcm(), rot2.as_dcm()))
    t_12 = rot1.apply(t2) + t1

    quat_res = rot_12.as_quat()
    return np.block([fromqxyzrToqrxyz(quat_res), t_12]) #np.block([ np.array([quat_res[3],quat_res[0],quat_res[1],quat_res[2]]), t_12])

''' q1 + q2, Quaternion '''
def composePoseQuat_array(q1, q2):
    if(q1.ndim == 2):
        res = np.empty((q1.shape[0],7))
        quat_arr_1 = fromqrxyzToqxyzr_array(q1)# np.array([q1[1], q1[2], q1[3], q1[0]])
        quat_arr_2 = fromqrxyzToqxyzr(q2)#np.array([q2[1], q2[2], q2[3], q2[0]])

        rot1 = scipy.spatial.transform.Rotation.from_quat(quat_arr_1)
        rot2 = scipy.spatial.transform.Rotation.from_quat(quat_arr_2)
        t1_array = q1[:,4:7]
        t2 = q2[4:7]

        rot_12_array = scipy.spatial.transform.Rotation.from_dcm(np.einsum('...ij,jk->...ik', rot1.as_dcm(), rot2.as_dcm()))
        res[:,0:4] = fromqxyzrToqrxyz_array(rot_12_array.as_quat())
        res[:,4:7] = np.einsum('...ij,j->...i', rot_12_array.as_dcm(),t2) + t1_array
    else:
        res = np.empty((q2.shape[0], 7))
        quat_arr_1 = fromqrxyzToqxyzr(q1)  # np.array([q1[1], q1[2], q1[3], q1[0]])
        quat_arr_2 = fromqrxyzToqxyzr_array(q2)  # np.array([q2[1], q2[2], q2[3], q2[0]])

        rot1 = scipy.spatial.transform.Rotation.from_quat(quat_arr_1)
        rot2 = scipy.spatial.transform.Rotation.from_quat(quat_arr_2)
        t1 = q1[4:7]
        t2 = q2[:,4:7]

        rot_12_array = scipy.spatial.transform.Rotation.from_dcm(np.einsum('ij,kjl->kil', rot1.as_dcm(), rot2.as_dcm()))
        res[:, 0:4] = fromqxyzrToqrxyz_array(rot_12_array.as_quat())
        res[:, 4:7] = rot1.apply(t2) + t1

    return res #np.block([fromqxyzrToqrxyz(quat_res), t_12]) #np.block([ np.array([quat_res[3],quat_res[0],quat_res[1],quat_res[2]]), t_12])

''' q + a, pose-point composition in Quaternion+3D '''
def composePoseQuatPoint(q_poseQuat, x):
    qx_sqr = q_poseQuat[1]**2
    qy_sqr = q_poseQuat[2]**2
    qz_sqr = q_poseQuat[3]**2
    qr_qx = q_poseQuat[0] * q_poseQuat[1]
    qr_qy = q_poseQuat[0] * q_poseQuat[2]
    qr_qz = q_poseQuat[0] * q_poseQuat[3]
    qx_qy = q_poseQuat[1] * q_poseQuat[2]
    qx_qz = q_poseQuat[1] * q_poseQuat[3]
    qy_qz = q_poseQuat[2] * q_poseQuat[3]
    ax = x[0]
    ay = x[1]
    az = x[2]
    point_composed = [ q_poseQuat[4] + ax + 2.*(-(qy_sqr + qz_sqr)*ax + (qx_qy - qr_qz)*ay + (qr_qy + qx_qz)*az),
                       q_poseQuat[5] + ay + 2.*( (qr_qz + qx_qy)*ax - (qx_sqr + qz_sqr)*ay + (qy_qz - qr_qx)*az),
                       q_poseQuat[6] + az + 2.*( (qx_qz - qr_qy)*ax + (qr_qx + qy_qz)*ay - (qx_sqr + qy_sqr)*az )
                     ]
    return point_composed

''' a - q '''
def inversePoseQuat_point(q_poseQuat, point):
    qx_sqr = q_poseQuat[1]**2
    qy_sqr = q_poseQuat[2]**2
    qz_sqr = q_poseQuat[3]**2
    qr_qx = q_poseQuat[0] * q_poseQuat[1]
    qr_qy = q_poseQuat[0] * q_poseQuat[2]
    qr_qz = q_poseQuat[0] * q_poseQuat[3]
    qx_qy = q_poseQuat[1] * q_poseQuat[2]
    qx_qz = q_poseQuat[1] * q_poseQuat[3]
    qy_qz = q_poseQuat[2] * q_poseQuat[3]
    dx = point[0] - q_poseQuat[4]
    dy = point[1] - q_poseQuat[5]
    dz = point[2] - q_poseQuat[6]
    a_composed = np.array([ dx + 2.*( -(qy_sqr + qz_sqr)*dx + (qx_qy + qr_qz)*dy + (-qr_qy + qx_qz)*dz ),
                   dy + 2.*( (-qr_qz + qx_qy)*dx - (qx_sqr + qz_sqr)*dy + (qy_qz + qr_qx)*dz ),
                   dz + 2.*( (qx_qz + qr_qy)*dx + (-qr_qx + qy_qz)*dy - (qx_sqr + qy_sqr)*dz )])
    return a_composed

def inversePoseQuat_point_array(q_poseQuat_array, point):
    qx_sqr = np.power(q_poseQuat_array[:,1],2)
    qy_sqr = np.power(q_poseQuat_array[:,2],2)
    qz_sqr = np.power(q_poseQuat_array[:,3],2)
    qr_qx = q_poseQuat_array[:,0] * q_poseQuat_array[:,1]
    qr_qy = q_poseQuat_array[:,0] * q_poseQuat_array[:,2]
    qr_qz = q_poseQuat_array[:,0] * q_poseQuat_array[:,3]
    qx_qy = q_poseQuat_array[:,1] * q_poseQuat_array[:,2]
    qx_qz = q_poseQuat_array[:,1] * q_poseQuat_array[:,3]
    qy_qz = q_poseQuat_array[:,2] * q_poseQuat_array[:,3]
    dx = point[0] - q_poseQuat_array[:,4]
    dy = point[1] - q_poseQuat_array[:,5]
    dz = point[2] - q_poseQuat_array[:,6]

    a_composed = np.empty((q_poseQuat_array.shape[0],3))
    a_composed[:,0] = dx + 2.*( -(qy_sqr + qz_sqr)*dx + (qx_qy + qr_qz)*dy + (-qr_qy + qx_qz)*dz )
    a_composed[:,1] = dy + 2.*( (-qr_qz + qx_qy)*dx - (qx_sqr + qz_sqr)*dy + (qy_qz + qr_qx)*dz )
    a_composed[:,2] = dz + 2.*( (qx_qz + qr_qy)*dx + (-qr_qx + qy_qz)*dy - (qx_sqr + qy_sqr)*dz )

    return a_composed

''' -q in Quaternion+3D '''
def inversePoseQuat(q_poseQuat):
    t = inversePoseQuat_point(q_poseQuat,[0.,0.,0.])
    return np.block([np.array([q_poseQuat[0], -q_poseQuat[1], -q_poseQuat[2], -q_poseQuat[3]]), t])#np.array([quat_inv[3], quat_inv[2,quat_inv[1],quat_inv[2]])

''' -q in Quaternion+3D '''
def inversePoseQuat_array(q_poseQuat_array):
    t_array = inversePoseQuat_point_array(q_poseQuat_array, [0.,0.,0.])
    res = np.empty((q_poseQuat_array.shape[0],4))
    res[:,0] = q_poseQuat_array[:,0]
    res[:,1] = -q_poseQuat_array[:,1]
    res[:,2] = -q_poseQuat_array[:,2]
    res[:,3] = -q_poseQuat_array[:,3]
    res[:,4] = t_array
    return res #np.array([quat_inv[3], quat_inv[2,quat_inv[1],quat_inv[2]])

''' Jacobian of quaternion normalization '''
def jacobianQuatNormalization(quat):
    qr_sqr = quat[0] ** 2
    qx_sqr = quat[1] ** 2
    qy_sqr = quat[2] ** 2
    qz_sqr = quat[3] ** 2
    qx_qr = quat[1] * quat[0]
    qy_qr = quat[2] * quat[0]
    qz_qr = quat[3] * quat[0]
    qx_qy = quat[1] * quat[2]
    qx_qz = quat[1] * quat[3]
    qy_qz = quat[2] * quat[3]

    K = 1./np.linalg.norm(quat[0:4])**3
    jacobian_normalization = np.array([ [qx_sqr + qy_sqr + qz_sqr, -qx_qr, -qy_qr, -qz_qr],
                                 [-qx_qr, qr_sqr + qy_sqr + qz_sqr, -qx_qy, -qx_qz],
                                 [-qy_qr, -qx_qy, qr_sqr + qx_sqr + qz_sqr, -qy_qz],
                                 [-qz_qr, -qx_qz, -qy_qz, qr_sqr + qx_sqr + qy_sqr]
                               ])
    return K*jacobian_normalization

def jacobianQuatNormalization_array(quat_array):
    qr_sqr = np.power(quat_array[:,0], 2)
    qx_sqr = np.power(quat_array[:,1], 2)
    qy_sqr = np.power(quat_array[:,2], 2)
    qz_sqr = np.power(quat_array[:,3], 2)
    qx_qr = quat_array[:,1] * quat_array[:,0]
    qy_qr = quat_array[:,2] * quat_array[:,0]
    qz_qr = quat_array[:,3] * quat_array[:,0]
    qx_qy = quat_array[:,1] * quat_array[:,2]
    qx_qz = quat_array[:,1] * quat_array[:,3]
    qy_qz = quat_array[:,2] * quat_array[:,3]

    K = 1./np.power(np.linalg.norm(quat_array[:,0:4],axis=1), 3)
    jacobian_normalization = np.empty((quat_array.shape[0],4,4))
    jacobian_normalization[:,0,0] = qx_sqr + qy_sqr + qz_sqr
    jacobian_normalization[:,0,1] = -qx_qr
    jacobian_normalization[:,0,2] = -qy_qr
    jacobian_normalization[:,0,3] = -qz_qr
    jacobian_normalization[:,1,0] = -qx_qr
    jacobian_normalization[:,1,1] = qr_sqr + qy_sqr + qz_sqr
    jacobian_normalization[:,1,2] = -qx_qy
    jacobian_normalization[:,1,3] = -qx_qz
    jacobian_normalization[:,2,0] = -qy_qr
    jacobian_normalization[:,2,1] = -qx_qy
    jacobian_normalization[:,2,2] =  qr_sqr + qx_sqr + qz_sqr
    jacobian_normalization[:,2,3] = -qy_qz
    jacobian_normalization[:,3,0] = -qz_qr
    jacobian_normalization[:,3,1] = -qx_qz
    jacobian_normalization[:,3,2] = -qy_qz
    jacobian_normalization[:,3,3] = qr_sqr + qx_sqr + qy_sqr

    return  K[:,None,None]*jacobian_normalization

''' J_q+a -> Jacobian of pose-point composition in quaternion '''
def computeJacobianQuat_composePosePoint(quat, a):
    qr_ax = quat[0]*a[0]
    qr_ay = quat[0]*a[1]
    qr_az = quat[0]*a[2]
    qx_ax = quat[1] * a[0]
    qx_ay = quat[1] * a[1]
    qx_az = quat[1] * a[2]
    qy_ax = quat[2] * a[0]
    qy_ay = quat[2] * a[1]
    qy_az = quat[2] * a[2]
    qz_ax = quat[3] * a[0]
    qz_ay = quat[3] * a[1]
    qz_az = quat[3] * a[2]

    A = np.array([[-qz_ay + qy_az, qy_ay + qz_az, -2.*qy_ax + qx_ay + qr_az, -2.*qz_ax - qr_ay + qx_az],
         [qz_ax - qx_az, qy_ax - 2.*qx_ay - qr_az, qx_ax + qz_az, qr_ax -2.*qz_ay + qy_az],
         [-qy_ax + qx_ay, qz_ax + qr_ay - 2.*qx_az, -qr_ax + qz_ay - 2.*qy_az, qx_ax + qy_ay]
        ])

    qr_sqr = quat[0]**2
    qx_sqr = quat[1]**2
    qy_sqr = quat[2]**2
    qz_sqr = quat[3]**2
    qx_qr = quat[1]*quat[0]
    qy_qr = quat[2]*quat[0]
    qz_qr = quat[3]*quat[0]
    qx_qy = quat[1]*quat[2]
    qx_qz = quat[1]*quat[3]
    qy_qz = quat[2]*quat[3]

    K = 1./np.linalg.norm(quat[0:4])**3
    jacobian_normalization = K*np.array([ [qx_sqr + qy_sqr + qz_sqr, -qx_qr, -qy_qr, -qz_qr],
                                 [-qx_qr, qr_sqr + qy_sqr + qz_sqr, -qx_qy, -qx_qz],
                                 [-qy_qr, -qx_qy, qr_sqr + qx_sqr + qz_sqr, -qy_qz],
                                 [-qz_qr, -qx_qz, -qy_qz, qr_sqr + qx_sqr + qy_sqr]
                               ])

    jacobianQuat_composePosePoint_pose_quat = 2.*np.matmul(A,jacobian_normalization)
    jacobianQuat_composePosePoint_pose = np.block([jacobianQuat_composePosePoint_pose_quat,np.eye(3)])

    jacobianQuat_composePosePoint_point = 2.*np.array([ [0.5 - qy_sqr - qz_sqr, qx_qy - qz_qr, qy_qr + qx_qz],
                                               [qz_qr + qx_qy, 0.5 - qx_sqr - qz_sqr, qy_qz - qx_qr],
                                               [qx_qz - qy_qr, qx_qr + qy_qz, 0.5 - qx_sqr - qy_sqr]
                                             ])

    return jacobianQuat_composePosePoint_pose, jacobianQuat_composePosePoint_point

''' J_q+a -> Jacobian of pose-point composition in quaternion '''
def computeJacobianQuat_composePosePoint_array(quat, a):
    if(quat.ndim == 2):
        qr_ax = quat[:,0] * a[0]
        qr_ay = quat[:,0] * a[1]
        qr_az = quat[:,0] * a[2]
        qx_ax = quat[:,1] * a[0]
        qx_ay = quat[:,1] * a[1]
        qx_az = quat[:,1] * a[2]
        qy_ax = quat[:,2] * a[0]
        qy_ay = quat[:,2] * a[1]
        qy_az = quat[:,2] * a[2]
        qz_ax = quat[:,3] * a[0]
        qz_ay = quat[:,3] * a[1]
        qz_az = quat[:,3] * a[2]

        A = np.empty((quat.shape[0],3,4))
        A[:,0,0] = -qz_ay + qy_az
        A[:,0,1] = qy_ay + qz_az
        A[:,0,2] = -2.*qy_ax + qx_ay + qr_az
        A[:,0,3] = -2.*qz_ax - qr_ay + qx_az
        A[:,1,0] = qz_ax - qx_az
        A[:,1,1] = qy_ax - 2.*qx_ay - qr_az
        A[:,1,2] = qx_ax + qz_az
        A[:,1,3] = qr_ax -2.*qz_ay + qy_az
        A[:,2,0] = -qy_ax + qx_ay
        A[:,2,1] = qz_ax + qr_ay - 2.*qx_az
        A[:,2,2] = -qr_ax + qz_ay - 2.*qy_az
        A[:,2,3] =  qx_ax + qy_ay

        qr_sqr = np.power(quat[:,0], 2)
        qx_sqr = np.power(quat[:,1],2)
        qy_sqr = np.power(quat[:,2],2)
        qz_sqr = np.power(quat[:,3],2)
        qx_qr = quat[:,1]*quat[:,0]
        qy_qr = quat[:,2]*quat[:,0]
        qz_qr = quat[:,3]*quat[:,0]
        qx_qy = quat[:,1]*quat[:,2]
        qx_qz = quat[:,1]*quat[:,3]
        qy_qz = quat[:,2]*quat[:,3]

        K = 1. / np.power(linalg.norm(quat[:,0:4]), 3)
        jacobian_normalization = np.empty((quat.shape[0], 4, 4))
        jacobian_normalization[:, 0, 0] = qx_sqr + qy_sqr + qz_sqr
        jacobian_normalization[:, 0, 1] = -qx_qr
        jacobian_normalization[:, 0, 2] = -qy_qr
        jacobian_normalization[:, 0, 3] = -qz_qr
        jacobian_normalization[:, 1, 0] = -qx_qr
        jacobian_normalization[:, 1, 1] = qr_sqr + qy_sqr + qz_sqr
        jacobian_normalization[:, 1, 2] = -qx_qy
        jacobian_normalization[:, 1, 3] = -qx_qz
        jacobian_normalization[:, 2, 0] = -qy_qr
        jacobian_normalization[:, 2, 1] = -qx_qy
        jacobian_normalization[:, 2, 2] = qr_sqr + qx_sqr + qz_sqr
        jacobian_normalization[:, 2, 3] = -qy_qz
        jacobian_normalization[:, 3, 0] = -qz_qr
        jacobian_normalization[:, 3, 1] = -qx_qz
        jacobian_normalization[:, 3, 2] = -qy_qz
        jacobian_normalization[:, 3, 3] = qr_sqr + qx_sqr + qy_sqr

        jacobian_normalization = K[:,None,None]*jacobian_normalization #np.einsum('k,kij->kij', K, jacobian_normalization)

        jacobianQuat_composePosePoint_pose_quat_array = 2.*np.einsum('kij,kjl->kil',A,jacobian_normalization)
        jacobianQuat_composePosePoint_pose_array = np.empty((quat_array.shape[0],3,7))
        jacobianQuat_composePosePoint_pose_array[:,:,0:4] = jacobianQuat_composePosePoint_pose_quat_array
        ones = np.full(quat_array.shape[0],1.)
        jacobianQuat_composePosePoint_pose_array[:,0,4] = ones
        jacobianQuat_composePosePoint_pose_array[:,1,5] = ones
        jacobianQuat_composePosePoint_pose_array[:,2,6] = ones

        jacobianQuat_composePosePoint_point_array = np.empyt(quat.shape[0],3,3)
        jacobianQuat_composePosePoint_point_array[:,0,0] = 0.5 - qy_sqr - qz_sqr,
        jacobianQuat_composePosePoint_point_array[:,0,1] = qx_qy - qz_qr
        jacobianQuat_composePosePoint_point_array[:,0,2] = qy_qr + qx_qz
        jacobianQuat_composePosePoint_point_array[:,1,0] = qz_qr + qx_qy
        jacobianQuat_composePosePoint_point_array[:,1,1] = 0.5 - qx_sqr - qz_sqr
        jacobianQuat_composePosePoint_point_array[:,1,2] = qy_qz - qx_qr
        jacobianQuat_composePosePoint_point_array[:,2,0] = qx_qz - qy_qr
        jacobianQuat_composePosePoint_point_array[:,2,1] = qx_qr + qy_qz
        jacobianQuat_composePosePoint_point_array[:,2,2] = 0.5 - qx_sqr - qy_sqr

        return jacobianQuat_composePosePoint_pose_array, 2.*jacobianQuat_composePosePoint_point
    else:
        qr_ax = quat[0] * a[:,0]
        qr_ay = quat[0] * a[:,1]
        qr_az = quat[0] * a[:,2]
        qx_ax = quat[1] * a[:,0]
        qx_ay = quat[1] * a[:,1]
        qx_az = quat[1] * a[:,2]
        qy_ax = quat[2] * a[:,0]
        qy_ay = quat[2] * a[:,1]
        qy_az = quat[2] * a[:,2]
        qz_ax = quat[3] * a[:,0]
        qz_ay = quat[3] * a[:,1]
        qz_az = quat[3] * a[:,2]

        A = np.empty((a.shape[0], 3, 4))
        A[:, 0, 0] = -qz_ay + qy_az
        A[:, 0, 1] = qy_ay + qz_az
        A[:, 0, 2] = -2. * qy_ax + qx_ay + qr_az
        A[:, 0, 3] = -2. * qz_ax - qr_ay + qx_az
        A[:, 1, 0] = qz_ax - qx_az
        A[:, 1, 1] = qy_ax - 2. * qx_ay - qr_az
        A[:, 1, 2] = qx_ax + qz_az
        A[:, 1, 3] = qr_ax - 2. * qz_ay + qy_az
        A[:, 2, 0] = -qy_ax + qx_ay
        A[:, 2, 1] = qz_ax + qr_ay - 2. * qx_az
        A[:, 2, 2] = -qr_ax + qz_ay - 2. * qy_az
        A[:, 2, 3] = qx_ax + qy_ay

        qr_sqr = np.power(quat[0], 2)
        qx_sqr = np.power(quat[1], 2)
        qy_sqr = np.power(quat[2], 2)
        qz_sqr = np.power(quat[3], 2)
        qx_qr = quat[1] * quat[0]
        qy_qr = quat[2] * quat[0]
        qz_qr = quat[3] * quat[0]
        qx_qy = quat[1] * quat[2]
        qx_qz = quat[1] * quat[3]
        qy_qz = quat[2] * quat[3]

        K = 1. / np.linalg.norm(quat[0:4]) ** 3
        jacobian_normalization = K * np.array([[qx_sqr + qy_sqr + qz_sqr, -qx_qr, -qy_qr, -qz_qr],
                                               [-qx_qr, qr_sqr + qy_sqr + qz_sqr, -qx_qy, -qx_qz],
                                               [-qy_qr, -qx_qy, qr_sqr + qx_sqr + qz_sqr, -qy_qz],
                                               [-qz_qr, -qx_qz, -qy_qz, qr_sqr + qx_sqr + qy_sqr]
                                               ])

        jacobianQuat_composePosePoint_pose_quat_array = 2. * np.einsum('kij,jl->kil', A, jacobian_normalization)
        jacobianQuat_composePosePoint_pose_array = np.zeros((a.shape[0], 3, 7))
        jacobianQuat_composePosePoint_pose_array[:, :, 0:4] = jacobianQuat_composePosePoint_pose_quat_array
        ones = np.full(a.shape[0], 1.)
        jacobianQuat_composePosePoint_pose_array[:, 0, 4] = ones
        jacobianQuat_composePosePoint_pose_array[:, 1, 5] = ones
        jacobianQuat_composePosePoint_pose_array[:, 2, 6] = ones

        jacobianQuat_composePosePoint_point = 2. * np.array([[0.5 - qy_sqr - qz_sqr, qx_qy - qz_qr, qy_qr + qx_qz],
                                                             [qz_qr + qx_qy, 0.5 - qx_sqr - qz_sqr, qy_qz - qx_qr],
                                                             [qx_qz - qy_qr, qx_qr + qy_qz, 0.5 - qx_sqr - qy_sqr]
                                                             ])

        return jacobianQuat_composePosePoint_pose_array, jacobianQuat_composePosePoint_point

''' J_q1+q2 -> Jacobian of poses composition in quaternion '''
def computeJacobianQuat_composePose(q_poseQuat1, q_poseQuat2):
    q_poseQuat_compose = composePoseQuat(q_poseQuat1,q_poseQuat2)

    jacobianQuat_composePosePoint_pose, jacobianQuat_composePosePoint_point = computeJacobianQuat_composePosePoint(q_poseQuat1, q_poseQuat2[4:7])

    J_normalization = jacobianQuatNormalization(q_poseQuat_compose)
    ''' Jacobian of the quaternion part w.r.t to quaternion variables '''
    jacobian_quat_quat_q1 = np.array([ [q_poseQuat2[0] , -q_poseQuat2[1] , -q_poseQuat2[2] , -q_poseQuat2[3]],
                              [q_poseQuat2[1] , q_poseQuat2[0]  , q_poseQuat2[3]  , -q_poseQuat2[2]],
                              [q_poseQuat2[2] , -q_poseQuat2[3] , q_poseQuat2[0]  , q_poseQuat2[1]] ,
                              [q_poseQuat2[3] , q_poseQuat2[2]  , -q_poseQuat2[1] , q_poseQuat2[0]]
                            ])
    jacobian_quat_quat_q1 = np.matmul(J_normalization, jacobian_quat_quat_q1)

    jacobian_quat_quat_q2 = np.array([ [q_poseQuat1[0] , -q_poseQuat1[1] , -q_poseQuat1[2] , -q_poseQuat1[3]],
                              [q_poseQuat1[1] , q_poseQuat1[0]  , -q_poseQuat1[3] , q_poseQuat1[2]] ,
                              [q_poseQuat1[2] , q_poseQuat1[3]  , q_poseQuat1[0]  , -q_poseQuat1[1]],
                              [q_poseQuat1[3] , -q_poseQuat1[2] , q_poseQuat1[1]  , q_poseQuat1[0]]
                            ])
    jacobian_quat_quat_q2 = np.matmul(J_normalization, jacobian_quat_quat_q2)

    jacobian_q1 = np.block([[jacobian_quat_quat_q1, np.zeros((4,3))],
                             [jacobianQuat_composePosePoint_pose],
                           ])
    jacobian_q2 = np.block([[jacobian_quat_quat_q2, np.zeros((4,3))],
                            [np.zeros((3, 4)), jacobianQuat_composePosePoint_point]
                           ])

    return jacobian_q1, jacobian_q2

''' J_q1+q2 -> Jacobian of poses composition in quaternion '''
def computeJacobianQuat_composePose_array(q_poseQuat1, q_poseQuat2):
    q_poseQuat_compose = composePoseQuat_array(q_poseQuat1, q_poseQuat2)
    J_normalization_array = jacobianQuatNormalization_array(q_poseQuat_compose)

    if(q_poseQuat1.ndim == 2):
        jacobianQuat_composePosePoint_pose, jacobianQuat_composePosePoint_point = computeJacobianQuat_composePosePoint_array(q_poseQuat1, q_poseQuat2[4:7])


        ''' Jacobian of the quaternion part w.r.t to quaternion variables '''
        jacobian_quat_quat_q1 = np.array([ [q_poseQuat2[0] , -q_poseQuat2[1] , -q_poseQuat2[2] , -q_poseQuat2[3]],
                                  [q_poseQuat2[1] , q_poseQuat2[0]  , q_poseQuat2[3]  , -q_poseQuat2[2]],
                                  [q_poseQuat2[2] , -q_poseQuat2[3] , q_poseQuat2[0]  , q_poseQuat2[1]] ,
                                  [q_poseQuat2[3] , q_poseQuat2[2]  , -q_poseQuat2[1] , q_poseQuat2[0]]
                                ])
        jacobian_quat_quat_q1_array = np.eisum('kij,jl->kil', J_normalization, jacobian_quat_quat_q1, optimize=True)
        jacobian_quat_quat_q2_array = np.empty((q_poseQuat1_array.shape[0],4,4))
        jacobian_quat_quat_q2_array[:,0,0] = q_poseQuat1[:,0]
        jacobian_quat_quat_q2_array[:,0,1] = -q_poseQuat1[:,1]
        jacobian_quat_quat_q2_array[:,0,2] = -q_poseQuat1[:,2]
        jacobian_quat_quat_q2_array[:,0,3] = -q_poseQuat1[:,3]
        jacobian_quat_quat_q2_array[:,1,0] = q_poseQuat1[:,1]
        jacobian_quat_quat_q2_array[:,1,1] = q_poseQuat1[:,0]
        jacobian_quat_quat_q2_array[:,1,2] = -q_poseQuat1[:,3]
        jacobian_quat_quat_q2_array[:,1,3] = q_poseQuat1[:,2]
        jacobian_quat_quat_q2_array[:,2,0] = q_poseQuat1[:,2]
        jacobian_quat_quat_q2_array[:,2,1] = q_poseQuat1[:,3]
        jacobian_quat_quat_q2_array[:,2,2] = q_poseQuat1[:,0]
        jacobian_quat_quat_q2_array[:,2,3] = -q_poseQuat1[:,1]
        jacobian_quat_quat_q2_array[:,3,0] = q_poseQuat1[:,3]
        jacobian_quat_quat_q2_array[:,3,1] = q_poseQuat1[:,2]
        jacobian_quat_quat_q2_array[:,3,2] = -q_poseQuat1[:,1]
        jacobian_quat_quat_q2_array[:,3,3] = q_poseQuat1[:,0]


        jacobian_quat_quat_q2_array = np.einsum('kij,kjl->kil', J_normalization_array, jacobian_quat_quat_q2_array)

        jacobian_q1 = np.zeros((q_poseQuat1_array.shape[0],7,7))
        jacobian_q1[:,4,4] = jacobian_quat_quat_q1_array
        jacobian_q1[:,3,7] = jacobianQuat_composePosePoint_pose

        jacobian_q2 = np.zeros((q_poseQuat1_array.shape[0],7,7))
        jacobian_q2[:,4,4] = jacobian_quat_quat_q2_array
        jacobian_q2[:,3,3] = jacobianQuat_composePosePoint_point

        return jacobian_q1, jacobian_q2
    else:
        jacobianQuat_composePosePoint_pose, jacobianQuat_composePosePoint_point = computeJacobianQuat_composePosePoint_array(q_poseQuat1, q_poseQuat2[:,4:7])

        ''' Jacobian of the quaternion part w.r.t to quaternion variables '''
        jacobian_quat_quat_q1 = np.empty((q_poseQuat2.shape[0],4,4))
        jacobian_quat_quat_q1[:,0,0] = q_poseQuat2[:,0]
        jacobian_quat_quat_q1[:,0,1] = -q_poseQuat2[:,1]
        jacobian_quat_quat_q1[:,0,2] = -q_poseQuat2[:,2]
        jacobian_quat_quat_q1[:,0,3] = -q_poseQuat2[:,3]
        jacobian_quat_quat_q1[:,1,0] = q_poseQuat2[:,1]
        jacobian_quat_quat_q1[:,1,1] = q_poseQuat2[:,0]
        jacobian_quat_quat_q1[:,1,2] = q_poseQuat2[:,3]
        jacobian_quat_quat_q1[:,1,3] = -q_poseQuat2[:,2]
        jacobian_quat_quat_q1[:,2,0] = q_poseQuat2[:,2]
        jacobian_quat_quat_q1[:,2,1] = -q_poseQuat2[:,3]
        jacobian_quat_quat_q1[:,2,2] = q_poseQuat2[:,0]
        jacobian_quat_quat_q1[:,2,3] = q_poseQuat2[:,1]
        jacobian_quat_quat_q1[:,3,0] = q_poseQuat2[:,3]
        jacobian_quat_quat_q1[:,3,1] = q_poseQuat2[:,2]
        jacobian_quat_quat_q1[:,3,2] = -q_poseQuat2[:,1]
        jacobian_quat_quat_q1[:,3,3] = q_poseQuat2[:,0]
        jacobian_quat_quat_q1= np.einsum('kij,kjl->kil', J_normalization_array, jacobian_quat_quat_q1, optimize=True)


        jacobian_quat_quat_q2 = np.array([[q_poseQuat1[0], -q_poseQuat1[1], -q_poseQuat1[2], -q_poseQuat1[3]],
                                          [q_poseQuat1[1], q_poseQuat1[0], -q_poseQuat1[3], q_poseQuat1[2]],
                                          [q_poseQuat1[2], q_poseQuat1[3], q_poseQuat1[0], -q_poseQuat1[1]],
                                          [q_poseQuat1[3], -q_poseQuat1[2], q_poseQuat1[1], q_poseQuat1[0]]
                                          ])

        jacobian_quat_quat_q2 = np.einsum('kij,jl->kil', J_normalization_array, jacobian_quat_quat_q2, optimize=True)

    jacobian_q1 = np.zeros((q_poseQuat2.shape[0], 7, 7))
    jacobian_q1[:, 0:4, 0:4] = jacobian_quat_quat_q1
    jacobian_q1[:, 4:7, 0:7] = jacobianQuat_composePosePoint_pose

    jacobian_q2 = np.zeros((q_poseQuat2.shape[0], 7, 7))
    jacobian_q2[:, 0:4, 0:4] = jacobian_quat_quat_q2
    for k in range(0,q_poseQuat2.shape[0]):
        jacobian_q2[k, 4:7, 4:7] = jacobianQuat_composePosePoint_point

    return jacobian_q1, jacobian_q2

''' J_a-q -> Jacobian of inverse pose-point composition in Quaternion+3D '''
def computeJacobian_inversePoseQuatPoint_pose(q_poseQuat, point):
    qx_sqr = q_poseQuat[1]**2
    qy_sqr = q_poseQuat[2]**2
    qz_sqr = q_poseQuat[3]**2
    qr_qx = q_poseQuat[0] * q_poseQuat[1]
    qr_qy = q_poseQuat[0] * q_poseQuat[2]
    qr_qz = q_poseQuat[0] * q_poseQuat[3]
    qx_qy = q_poseQuat[1] * q_poseQuat[2]
    qx_qz = q_poseQuat[1] * q_poseQuat[3]
    qy_qz = q_poseQuat[2] * q_poseQuat[3]
    dx = point[0] - q_poseQuat[4]
    dy = point[1] - q_poseQuat[5]
    dz = point[2] - q_poseQuat[6]
    qr_dx = q_poseQuat[0] * dx
    qr_dy = q_poseQuat[0] * dy
    qr_dz = q_poseQuat[0] * dz
    qx_dx = q_poseQuat[1] * dx
    qx_dy = q_poseQuat[1] * dy
    qx_dz = q_poseQuat[1] * dz
    qy_dx = q_poseQuat[2] * dx
    qy_dy = q_poseQuat[2] * dy
    qy_dz = q_poseQuat[2] * dz
    qz_dx = q_poseQuat[3] * dx
    qz_dy = q_poseQuat[3] * dy
    qz_dz = q_poseQuat[3] * dz

    J_trans = 2.*np.array([[qy_sqr + qz_sqr - 0.5, -(qr_qz + qx_qy), qr_qy - qx_qz],
                           [qr_qz - qx_qy, qx_sqr + qz_sqr - 0.5, -(qr_qx + qy_qz)],
                           [-(qr_qy + qx_qz), qr_qx - qy_qz, qx_sqr + qy_sqr -0.5]
                          ])
    J_rot_left = 2.*np.array([[-qy_dz + qz_dy, qy_dy + qz_dz, qx_dy - 2.*qy_dx - qr_dz, qx_dz + qr_dy -2.*qz_dx],
                         [qx_dz - qz_dx, qy_dx -2.*qx_dy + qr_dz, qx_dx + qz_dz, -qr_dx - 2.*qz_dy + qy_dz],
                         [qy_dx - qx_dy, qz_dx - qr_dy - 2.*qx_dz, qz_dy + qr_dx - 2.*qy_dz, qx_dx + qy_dy]
                         ])

    J_norm = jacobianQuatNormalization(q_poseQuat)
    J_rot = np.matmul(J_rot_left,J_norm)
    return np.block([J_rot, J_trans])

''' J_a-q -> Jacobian of inverse pose-point composition in Quaternion+3D '''
def computeJacobian_inversePoseQuatPoint_pose_array(q_poseQuat_array, point):
    qx_sqr = np.power(q_poseQuat_array[:, 1], 2)
    qy_sqr = np.power(q_poseQuat_array[:, 2], 2)
    qz_sqr = np.power(q_poseQuat_array[:, 3], 2)
    qr_qx = q_poseQuat_array[:,0] * q_poseQuat_array[:,1]
    qr_qy = q_poseQuat_array[:,0] * q_poseQuat_array[:,2]
    qr_qz = q_poseQuat_array[:,0] * q_poseQuat_array[:,3]
    qx_qy = q_poseQuat_array[:,1] * q_poseQuat_array[:,2]
    qx_qz = q_poseQuat_array[:,1] * q_poseQuat_array[:,3]
    qy_qz = q_poseQuat_array[:,2] * q_poseQuat_array[:,3]
    dx = point[0] - q_poseQuat_array[:,4]
    dy = point[1] - q_poseQuat_array[:,5]
    dz = point[2] - q_poseQuat_array[:,6]
    qr_dx = q_poseQuat_array[:,0] * dx
    qr_dy = q_poseQuat_array[:,0] * dy
    qr_dz = q_poseQuat_array[:,0] * dz
    qx_dx = q_poseQuat_array[:,1] * dx
    qx_dy = q_poseQuat_array[:,1] * dy
    qx_dz = q_poseQuat_array[:,1] * dz
    qy_dx = q_poseQuat_array[:,2] * dx
    qy_dy = q_poseQuat_array[:,2] * dy
    qy_dz = q_poseQuat_array[:,2] * dz
    qz_dx = q_poseQuat_array[:,3] * dx
    qz_dy = q_poseQuat_array[:,3] * dy
    qz_dz = q_poseQuat_array[:,3] * dz

    J_trans = np.empty((q_poseQuat_array.shape[0],3,3))
    J_trans[:,0,0] = qy_sqr + qz_sqr - 0.5
    J_trans[:,0,1] = -(qr_qz + qx_qy)
    J_trans[:,0,2] = qr_qy - qx_qz
    J_trans[:,1:0] = qr_qz - qx_qy
    J_trans[:,1,1] = qx_sqr + qz_sqr - 0.5
    J_trans[:,1,2] = -(qr_qx + qy_qz)
    J_trans[:,2,0] = -(qr_qy + qx_qz)
    J_trans[:,2,1] = qr_qx - qy_qz
    J_trans[:,2,2] = qx_sqr + qy_sqr -0.5

    J_rot_left = np.empty((q_poseQuat_array.shape[0],3,4))
    J_rot_left[:,0,0] = -qy_dz + qz_dy
    J_rot_left[:,0,1] = qy_dy + qz_dz
    J_rot_left[:,0,2] = qx_dy - 2.*qy_dx - qr_dz
    J_rot_left[:,0,3] = qx_dz + qr_dy -2.*qz_dx
    J_rot_left[:,1,0] = qx_dz - qz_dx
    J_rot_left[:,1,1] = qy_dx -2.*qx_dy + qr_dz
    J_rot_left[:,1,2] = qx_dx + qz_dz
    J_rot_left[:,1,3] = -qr_dx - 2.*qz_dy + qy_dz
    J_rot_left[:,2,0] = qy_dx - qx_dy
    J_rot_left[:,2,1] = qz_dx - qr_dy - 2.*qx_dz
    J_rot_left[:,2,2] = qz_dy + qr_dx - 2.*qy_dz
    J_rot_left[:,2,3] = qx_dx + qy_dy

    J_norm = jacobianQuatNormalization_array(q_poseQuat_array)
    J_rot = np.einsum('kij, kjl->kil',2.*J_rot_left, J_norm) #np.matmul(J_rot_left, J_norm)

    J_final = np.empty((q_poseQuat_array.shape[0],3,7))
    J_final[:,:,0:4] = J_rot
    J_final[:,:,4:7] = 2.*J_trans
    return J_final

''' J_-q -> Jacobian of inverse pose in Quaternion+3D '''
def computeJacobian_inversePoseQuat(q_poseQuat):
    A = -1*np.eye(4)
    A[0][0] = 1
    J_rot = np.matmul(A, jacobianQuatNormalization(q_poseQuat))
    J_sub = computeJacobian_inversePoseQuatPoint_pose(q_poseQuat, [0.,0.,0.])
    J = np.block([[J_rot, np.zeros((4,3))],
                  [J_sub],
                 ])
    return J

''' J_-q -> Jacobian of inverse pose in Quaternion+3D '''
def computeJacobian_inversePoseQuat_array(q_poseQuat_array):
    A = -1*np.eye(4)
    A[0][0] = 1
    #J_rot = np.einsum('ij,kjl->kil', A, jacobianQuatNormalization_array(q_poseQuat_array)) #np.matmul(A, jacobianQuatNormalization_array(q_poseQuat_array))
    #J_sub = computeJacobian_inversePoseQuatPoint_pose_array(q_poseQuat_array, [0.,0.,0.])

    J = np.zeros((q_poseQuat_array.shape[0], 7, 7))
    J[:, 0:4, 0:4] = np.einsum('ij,kjl->kil', A, jacobianQuatNormalization_array(q_poseQuat_array))
    J[:, 4:7, 0:7] = computeJacobian_inversePoseQuatPoint_pose_array(q_poseQuat_array, [0.,0.,0.])

    return J

''' -q in Quaternion+3D PDF '''
def inversePosePDFQuat_array(q_posePDFQuat_array):
    inverse_mean_array = inversePoseQuat_array(q_posePDFQuat_array['pose_mean'])
    J = computeJacobian_inversePoseQuat_array(q_posePDFQuat_array['pose_mean'])
    inverse_cov_array = np.einsum('kij,kjl->kil', J, np.einsum('kij,klj->kil',q_posePDFQuat_array['pose_cov'],J,optimize=True),optimize=True)  #np.matmul(J, np.matmul(q_posePDFQuat_array['pose_cov'], np.transpose(J)))
    return {'pose_mean' : inverse_mean_array, 'pose_cov': inverse_cov_array}

''' -q in Quaternion+3D PDF '''
def inversePosePDFQuat(q_posePDFQuat):
    inverse_mean = inversePoseQuat(q_posePDFQuat['pose_mean'])
    J = computeJacobian_inversePoseQuat(q_posePDFQuat['pose_mean'])
    inverse_cov = np.matmul(J, np.matmul(q_posePDFQuat['pose_cov'], np.transpose(J)))
    return {'pose_mean' : inverse_mean, 'pose_cov': inverse_cov}

''' q1 + q2, in Quaternion+3D PDF  '''
def composePosePDFQuat(q_posePDFQuat1, q_posePDFQuat2):
    q_poseQuat_composed_mean = composePoseQuat(q_posePDFQuat1['pose_mean'], q_posePDFQuat2['pose_mean'])
    jacobian_quat_q1, jacobian_quat_q2 = computeJacobianQuat_composePose(q_posePDFQuat1['pose_mean'], q_posePDFQuat2['pose_mean'])
    q_poseQuat_composed_cov = np.matmul(jacobian_quat_q1, np.matmul(q_posePDFQuat1['pose_cov'], np.transpose(jacobian_quat_q1))) + \
                              np.matmul(jacobian_quat_q2, np.matmul(q_posePDFQuat2['pose_cov'], np.transpose(jacobian_quat_q2)))

    return {'pose_mean' : q_poseQuat_composed_mean, 'pose_cov' : q_poseQuat_composed_cov}

''' q1 + q2, in Quaternion+3D PDF  '''
def composePosePDFQuat_array(q_posePDFQuat1, q_posePDFQuat2):
    q_poseQuat_composed_mean = composePoseQuat_array(q_posePDFQuat1['pose_mean'], q_posePDFQuat2['pose_mean'])
    jacobian_quat_q1, jacobian_quat_q2 = computeJacobianQuat_composePose_array(q_posePDFQuat1['pose_mean'], q_posePDFQuat2['pose_mean'])
    q_poseQuat_composed_cov = np.einsum('...ij,...jl->...il', jacobian_quat_q1, np.einsum('...ij,...lj->...il',q_posePDFQuat1['pose_cov'],jacobian_quat_q1,optimize=True),optimize=True) + \
                              np.einsum('kij,kjl->kil', jacobian_quat_q2, np.einsum('kij,klj->kil', q_posePDFQuat2['pose_cov'],jacobian_quat_q2, optimize=True), optimize=True)


    return {'pose_mean': q_poseQuat_composed_mean, 'pose_cov': q_poseQuat_composed_cov}

def composePosePDFQuatPoint(q_posePDFQuat, x):
    rot = scipy.spatial.transform.Rotation.from_quat(fromqrxyzToqxyzr(q_posePDFQuat["pose_mean"][0:4]))
    point_composed_mean = rot.apply(x["mean"]) + q_posePDFQuat["pose_mean"][4:7]

    jacobianQuat_composePosePoint_pose, jacobianQuat_composePosePoint_point = computeJacobianQuat_composePosePoint_array(q_posePDFQuat["pose_mean"], x["mean"])
    point_composed_cov = np.einsum('kij,kjl->kil', jacobianQuat_composePosePoint_pose, np.einsum('ij,klj->kil',q_posePDFQuat['pose_cov'],jacobianQuat_composePosePoint_pose,optimize=True),optimize=True) + \
                    np.einsum('ij,kjl->kil', jacobianQuat_composePosePoint_point,np.einsum('kij,lj->kil', x['cov'], jacobianQuat_composePosePoint_point,
                              optimize=True), optimize=True)
    return {"mean": point_composed_mean, "cov": point_composed_cov}

def inverseComposePoseQuat(q_poseQuat1, q_poseQuat2):
    q_poseQuat2_inv = inversePoseQuat(q_poseQuat2)
    return composePoseQuat(q_poseQuat2_inv, q_poseQuat1)

''' q1 - q2 with uncertainty '''
def inverseComposePosePDFQuat(q_posePDFQuat1, q_posePDFQuat2):
    ''' q1 - q2 =  (-q2) + q1 '''
    q_posePDFQuat2_inv = inversePosePDFQuat(q_posePDFQuat2)
    return composePosePDFQuat(q_posePDFQuat2_inv, q_posePDFQuat1)

''' q1 - q2 with uncertainty '''
def inverseComposePosePDFQuat_array(q_posePDFQuat1_array, q_posePDFQuat2):
    ''' q1 - q2 = (-q2) + q1 '''
    q_posePDFQuat2_inv = inversePosePDFQuat(q_posePDFQuat2)
    return composePosePDFQuat_array(q_posePDFQuat2_inv, q_posePDFQuat1_array)

def distanceSE3(p1, p2):
    return math_utility.distanceSE3(poses_euler.fromPoseQuatToPoseEuler(p1), poses_euler.fromPoseQuatToPoseEuler(p2))
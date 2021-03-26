''' Operations and functions related to SE(3) poses
    Implementation based on :
    "J.-L. Blanco. A tutorial on se (3) transformation parameterizations and
     on-manifold optimization. University of Malaga, Tech. Rep, 3, 2010."
'''

import math_utility
import numba as nb
import poses_quat
import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats

''' q1 + q2 , Euler'''
def composePoseEuler(q1, q2):
    rot1 = scipy.spatial.transform.Rotation.from_euler('ZYX',q1[0:3])
    rot2 = scipy.spatial.transform.Rotation.from_euler('ZYX',q2[0:3])
    t1 = q1[3:6]
    t2 = q2[3:6]

    rot_12 = scipy.spatial.transform.Rotation.from_matrix(np.matmul(rot1.as_matrix(), rot2.as_matrix()))
    t_12 = rot1.apply(t2) + t1

    return np.block([rot_12.as_euler('ZYX'), t_12])

def composePoseEuler_array(q1, q2_array):
    res = np.empty((q2_array.shape[0],6))
    rot1 = scipy.spatial.transform.Rotation.from_euler('ZYX',q1[0:3])
    rot2 = scipy.spatial.transform.Rotation.from_euler('ZYX',q2_array[:,0:3])
    t1 = q1[3:6]
    t2 = q2_array[:,3:6]

    rot_12 = scipy.spatial.transform.Rotation.from_dcm(np.einsum('ij,kjl->kil', rot1.as_dcm(), rot2.as_dcm())) #np.matmul(rot1.as_dcm(), rot2.as_dcm()))
    res[:,0:3] = scipy.spatial.transform.Rotation.from_dcm(np.einsum('ij,kjl->kil', rot1.as_dcm(), rot2.as_dcm())).as_euler('ZYX')
    res[:,3:6] = rot1.apply(t2) + t1

    return res

''' q + a , pose-point composition in Euler+3D '''
def composePoseEulerPoint(q_poseEuler, x):
    rot = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[0:3])
    t = q_poseEuler[3:6]
    x_composed = rot.apply(x) + t
    return x_composed

''' q1 - q2  = -q2 + q1 in Euler+3D '''
def inverseComposePoseEuler(q_poseEuler1, q_poseEuler2):
    rot1 = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler1[0:3])
    rot2 = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler2[0:3]).inv()
    t1 = q_poseEuler1[3:6]
    t2 = -rot2.apply(q_poseEuler2[3:6])

    rot_12 = scipy.spatial.transform.Rotation.from_dcm(np.matmul(rot1.as_dcm(), rot2.as_dcm()))
    t_12 = rot1.apply(t2) + t1

    return np.block([rot_12.as_euler('ZYX'), t_12])

''' q1 - q2 in Euler+3D with q1 as a 2D array (stack of euler angles) ie res = [q_i - q2 ]'''
def inverseComposePoseEuler_array(q_poseEuler1_array, q_poseEuler2):
    res = np.empty((q_poseEuler1_array.shape[0],6))
    rot1 = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler1_array[:,0:3])
    rot2 = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler2[0:3]).inv()
    t1 = q_poseEuler1_array[:,3:6]
    t2 = -rot2.apply(q_poseEuler2[3:6])

    rot1_mat = rot1.as_dcm()
    res[:,0:3] = scipy.spatial.transform.Rotation.from_dcm(np.einsum('...ij,jk->...ik', rot1_mat, rot2.as_dcm())).as_euler('ZYX')
    res[:,3:6] = np.einsum('...ij,j->...i', rot1_mat, t2) + t1

    return res

''' a - q, inverse pose-point composition in Euler+3D '''
def inverseComposePoseEulerPoint(q_poseEuler, x):
    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[0:3]).inv()
    t = q_poseEuler[3:6]
    #diff = []
    #for x_ in x:
    #    diff.append(x_ - t)
    x_composed = rot_T.apply(x - t)#(np.array(diff))
    return x_composed

''' a - q, inverse pose-point composition in Euler+3D '''
def inverseComposePoseEulerPoint_opt(q_poseEuler, x):
    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[:,0:3]).inv()
    t = q_poseEuler[:,3:6]
    x_composed = rot_T.apply(x - t)
    return x_composed

def inverseComposePoseEulerPoint_opt_array(q_poseEuler, x_array):
    ''' Reshape the n_samples X n_points X 6 3D array of q_poseEuler to n_samples * n_points X 6 2D array '''
    #''' Reshape the n_samples X 6 X n_points 3D array of q_poseEuler to n_samples * n_points X 6 2D array '''
    n_samples = q_poseEuler.shape[0]
    n_points = q_poseEuler.shape[1]
    q_posesEuler_reshaped = q_poseEuler.reshape((n_samples * n_points, 6), order='F')
    #q_posesEuler_reshaped = q_poseEuler.reshape((n_samples * n_points, 6), order='C')

    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posesEuler_reshaped[:, 0:3]).inv()
    t = q_poseEuler[:,:,3:6]

    ''' Dimension : n_samples * n_points X 3 X 1 '''
    diff = (x_array - t).reshape((n_samples*n_points,3,1),order='F')

    ''' Multiply slice by slice ie R_{i,j}(x_i - t_{i,j}) '''
    ''' Reshape the results back to a 3D array with samples as the depth dimension '''
    x_composed_array = np.einsum('...ij,...jh->...ih', rot_T.as_dcm(), diff,optimize=True).reshape((n_samples, n_points, 3),order='F')
    return x_composed_array

def inverseComposePoseEulerPoint_opt_array_association(q_poseEuler, x_array):
    ''' Reshape the n_samples X n_points X 6 3D array of q_poseEuler to n_samples * n_points X 6 2D array '''
    #''' Reshape the n_samples X 6 X n_points 3D array of q_poseEuler to n_samples * n_points X 6 2D array '''
    n_samples = q_poseEuler.shape[0]
    n_points_q = q_poseEuler.shape[1]
    n_points_x = x_array.shape[0]
    q_posesEuler_reshaped = q_poseEuler.reshape((n_samples * n_points_q, 6), order='C')
    #q_posesEuler_reshaped = q_poseEuler.reshape((n_samples * n_points, 6), order='C')

    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posesEuler_reshaped[:, 0:3]).inv()
    t = q_posesEuler_reshaped[:,3:6] #q_poseEuler[:,:,3:6]

    ''' Dimension of diff : n_samples X n_points_q X n_points_x (x_array) X 3 '''
    ''' We have diff[k][i][j] = x_array[j] - t[k][i] '''
    diff = (x_array - t[:,None]).reshape((n_samples,n_points_q,n_points_x, 3))
    #diff = (x_array - t).reshape((n_samples*n_points,3,1),order='F')

    ''' Multiply slice by slice ie R_{i,k}(x_j - t_{i,k}) '''
    #''' Reshape the results back to a 3D array with samples as the depth dimension '''
    x_composed_array = np.einsum('kilm,kijm->kijl', rot_T.as_dcm().reshape((n_samples,n_points_q,3,3)), diff,optimize=True)
    return x_composed_array

''' Convert Pose in Euler+3D to Pose in Quaternion+3D '''
def fromPoseEulerToPoseQuat(q_poseEuler):
    rot = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[0:3])
    quat = rot.as_quat()
    ''' numpy quaternion -> [qx qy qz qr] , mrpt quat is [qr qx qy qz] '''
    quat_arr = poses_quat.fromqxyzrToqrxyz(quat) #np.array([quat[3], quat[0], quat[1], quat[2]])
    q_poseQuat = np.block([quat_arr, np.array(q_poseEuler[3:6])])
    return q_poseQuat

''' Convert Pose in Euler+3D to Pose in Quaternion+3D '''
def fromPoseEulerToPoseQuat_array(q_poseEuler_array):
    rot = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler_array[:,0:3])
    ''' numpy quaternion -> [qx qy qz qr] , mrpt quat is [qr qx qy qz] '''
    #quat_arr = poses_quat.fromqxyzrToqrxyz_array(rot.as_quat()) #poses_quat.fromqxyzrToqrxyz(quat) #np.array([quat[3], quat[0], quat[1], quat[2]])
    q = rot.as_quat()

    q_poseQuat_array = np.empty((q_poseEuler_array.shape[0],7))
    q_poseQuat_array[:, 0] = q[:, 3]
    q_poseQuat_array[:, 1] = q[:, 0]
    q_poseQuat_array[:, 2] = q[:, 1]
    q_poseQuat_array[:, 3] = q[:, 2]
    q_poseQuat_array[:, 4:7] = q_poseEuler_array[:, 3:6]
    #q_poseQuat_array = np.block([quat_arr, np.array(q_poseEuler[3:6])])
    return q_poseQuat_array

''' Convert Pose in Quaternion+3D to Pose in Euler+3D '''
def fromPoseQuatToPoseEuler(q_poseQuat):
    quat = poses_quat.fromqrxyzToqxyzr(q_poseQuat) #np.array([q_poseQuat[1], q_poseQuat[2],q_poseQuat[3],q_poseQuat[0]])
    ''' Quaternion is normalized here '''
    rot = scipy.spatial.transform.Rotation.from_quat(quat)
    q_poseEuler = np.block([np.array(rot.as_euler('ZYX')), np.array(q_poseQuat[4:7])])
    return q_poseEuler

''' Convert Pose in Quaternion+3D to Pose in Euler+3D '''
def fromPoseQuatToPoseEuler_array(q_poseQuat_array):
    quat = poses_quat.fromqrxyzToqxyzr_array(q_poseQuat_array) #np.array([q_poseQuat[1], q_poseQuat[2],q_poseQuat[3],q_poseQuat[0]])
    ''' Quaternion is normalized here '''
    rot = scipy.spatial.transform.Rotation.from_quat(quat)

    q_poseEuler_array = np.empty((q_poseQuat_array.shape[0], 6))
    q_poseEuler_array[:,0:3] = rot.as_euler('ZYX')
    q_poseEuler_array[:,3:6] = q_poseQuat_array[:,4:7]

    return q_poseEuler_array

''' J_q->e -> Jacobian of convertion from Quaternion+3D to Euler+3D '''
def computeJacobian_quatToEuler(q_poseQuat):
    ''' Compute the jacobian of euler angles wrt to quaternion'''
    determinant = q_poseQuat[0]*q_poseQuat[2] - q_poseQuat[1]*q_poseQuat[3]
    jacobian_euler_quat = np.zeros((3, 4))
    if(determinant > 0.49999):
        num = 2./(q_poseQuat[0]**2 + q_poseQuat[1]**2)
        jacobian_euler_quat[0][0] = num*q_poseQuat[1]
        jacobian_euler_quat[0][2] = -num*q_poseQuat[0]
    elif(determinant < -0.49999):
        num = 2. / (q_poseQuat[0] ** 2 + q_poseQuat[1] ** 2)
        jacobian_euler_quat[0][0] = -num * q_poseQuat[1]
        jacobian_euler_quat[0][2] = num * q_poseQuat[0]
    else:
        x_sqr = q_poseQuat[1]**2
        y_sqr = q_poseQuat[2]**2
        z_sqr = q_poseQuat[3]**2
        r_z = q_poseQuat[0]*q_poseQuat[3]
        r_x = q_poseQuat[0]*q_poseQuat[1]
        x_y = q_poseQuat[1]*q_poseQuat[2]
        y_z = q_poseQuat[2]*q_poseQuat[3]
        a = 1. - 2.*(y_sqr + z_sqr)
        a_sqr = a**2
        b = 2.*(r_z + x_y)
        c = 1. - 2.*(x_sqr + y_sqr)
        c_sqr = c**2
        d = 2.*(r_x + y_z)

        atan_prime_yaw = 1./(1. + (b/a)**2)
        atan_prime_roll = 1./(1. + (d/c)**2)
        asin_prime = 1./np.sqrt(1. - 4.*determinant**2)

        jacobian_euler_quat[0][0] = 2.*q_poseQuat[3]*atan_prime_yaw/a
        jacobian_euler_quat[0][1] = 2.*q_poseQuat[2]*atan_prime_yaw/a
        jacobian_euler_quat[0][2] = 2.*((q_poseQuat[1]*a + 2.*q_poseQuat[2]*b)/a_sqr)*atan_prime_yaw
        jacobian_euler_quat[0][3] = 2.*((q_poseQuat[0]*a + 2.*q_poseQuat[3]*b)/a_sqr)*atan_prime_yaw

        jacobian_euler_quat[1][0] = 2.*q_poseQuat[2]*asin_prime
        jacobian_euler_quat[1][1] = -2. * q_poseQuat[3] * asin_prime
        jacobian_euler_quat[1][2] = 2. * q_poseQuat[0] * asin_prime
        jacobian_euler_quat[1][3] = -2. * q_poseQuat[1] * asin_prime

        jacobian_euler_quat[2][0] = 2.*(q_poseQuat[1]/c)*atan_prime_roll
        jacobian_euler_quat[2][1] = 2. * ((q_poseQuat[0]*c + 2.*q_poseQuat[1]*d)/c_sqr) * atan_prime_roll
        jacobian_euler_quat[2][2] = 2. * ((q_poseQuat[3]*c + 2.*q_poseQuat[2]*d)/c_sqr) * atan_prime_roll
        jacobian_euler_quat[2][3] = 2. * (q_poseQuat[2] / c) * atan_prime_roll

        J_norm = poses_quat.jacobianQuatNormalization(q_poseQuat[0:4])


    jacobian = np.block([[jacobian_euler_quat@J_norm , np.zeros((3,3))],
                         [np.zeros((3,4)), np.eye(3)]
                        ])

    return jacobian

''' J_q->e -> Jacobian of convertion from Quaternion+3D to Euler+3D '''
def computeJacobian_quatToEuler_array(q_poseQuat_array):
    ''' Compute the jacobian of euler angles wrt to quaternion'''
    determinant = q_poseQuat_array[:,0]*q_poseQuat_array[:,2] - q_poseQuat_array[:,1]*q_poseQuat_array[:,3]
    jacobian_euler_quat = np.zeros((q_poseQuat_array.shape[0],3, 4))

    num = 2./(np.power(q_poseQuat_array[:,0],2) + np.power(q_poseQuat_array[:,1],2))
    num_poseQuat0 = num*q_poseQuat_array[:,0]
    num_poseQuat1 = num * q_poseQuat_array[:, 1]

    x_sqr = np.power(q_poseQuat_array[:,1],2)
    y_sqr = np.power(q_poseQuat_array[:,2],2)
    z_sqr = np.power(q_poseQuat_array[:,3],2)
    r_z = q_poseQuat_array[:,0] * q_poseQuat_array[:,3]
    r_x = q_poseQuat_array[:,0] * q_poseQuat_array[:,1]
    x_y = q_poseQuat_array[:,1] * q_poseQuat_array[:,2]
    y_z = q_poseQuat_array[:,2] * q_poseQuat_array[:,3]
    a = 1. - 2. * (y_sqr + z_sqr)
    a_inv = 1./a
    a_sqr = np.power(a,2)
    a_sqr_inv = 1./a_sqr
    b = 2. * (r_z + x_y)
    c = 1. - 2. * (x_sqr + y_sqr)
    c_inv = 1./c
    c_sqr_inv = 1./np.power(c,2)
    d = 2. * (r_x + y_z)

    atan_prime_yaw = 1. / (1. + np.power(b*a_inv,2))
    atan_prime_roll = 1. / (1. + np.power(d*c_inv,2))
    asin_prime = 1. / np.sqrt(1. - 4. * np.power(determinant,2))

    a00 = 2.*q_poseQuat_array[:,3]*atan_prime_yaw*a_inv
    a01 = 2.*q_poseQuat_array[:,2]*atan_prime_yaw*a_inv
    a02 = 2.*((q_poseQuat_array[:,1]*a + 2.*q_poseQuat_array[:,2]*b)*a_sqr_inv)*atan_prime_yaw
    a03 = 2. * ((q_poseQuat_array[:,0] * a + 2. * q_poseQuat_array[:,3] * b)* a_sqr_inv) * atan_prime_yaw
    a10 = 2.*q_poseQuat_array[:,2]*asin_prime
    a11 = -2. * q_poseQuat_array[:,3] * asin_prime
    a12 = 2. * q_poseQuat_array[:,0] * asin_prime
    a13 = -2. * q_poseQuat_array[:,1] * asin_prime
    a20 = 2.*(q_poseQuat_array[:,1]*c_inv)*atan_prime_roll
    a21 = 2. * ((q_poseQuat_array[:,0]*c + 2.*q_poseQuat_array[:,1]*d)*c_sqr_inv) * atan_prime_roll
    a22 = 2. * ((q_poseQuat_array[:,3]*c + 2.*q_poseQuat_array[:,2]*d)*c_sqr_inv) * atan_prime_roll
    a23 = 2. * (q_poseQuat_array[:,2]*c_inv) * atan_prime_roll
    for k in range(0,determinant.shape[0]):
        det = determinant[k]
        if(det> 0.49999):
            jacobian_euler_quat[k,0,0] = num_poseQuat1[k]
            jacobian_euler_quat[k,0,2] = -num_poseQuat0[k]
        elif(det< -0.49999):
            jacobian_euler_quat[k,0,0] = -num_poseQuat1[k]
            jacobian_euler_quat[k,0,2] = num_poseQuat0[k]
        else:
            jacobian_euler_quat[k,0,0] = a00[k]
            jacobian_euler_quat[k,0,1] = a01[k]
            jacobian_euler_quat[k,0,2] = a02[k]
            jacobian_euler_quat[k,0,3] = a03[k]

            jacobian_euler_quat[k,1,0] = a10[k]
            jacobian_euler_quat[k,1,1] = a11[k]
            jacobian_euler_quat[k,1,2] = a12[k]
            jacobian_euler_quat[k,1,3] = a13[k]

            jacobian_euler_quat[k,2,0] = a20[k]
            jacobian_euler_quat[k,2,1] = a21[k]
            jacobian_euler_quat[k,2,2] = a22[k]
            jacobian_euler_quat[k,2,3] = a23[k]

    J_norm = poses_quat.jacobianQuatNormalization_array(q_poseQuat_array[:,0:4])

    jacobian = np.zeros((q_poseQuat_array.shape[0],6,7))
    jacobian[:,0:3,0:4] = np.einsum('kij,kjl->kil',jacobian_euler_quat,J_norm,optimize=True)
    ones = np.full(q_poseQuat_array.shape[0],1.)
    jacobian[:,3,4] = ones
    jacobian[:,4,5] = ones
    jacobian[:,5,6] = ones

    return jacobian

''' J_e->q -> Jacobian of convertion from Euler+3D to Quaternion+3D '''
def computeJacobian_eulerToQuat(q_poseEuler):
    half_yaw = 0.5*q_poseEuler[0]
    half_pitch = 0.5*q_poseEuler[1]
    half_roll = 0.5*q_poseEuler[2]
    cos_yaw = np.cos(half_yaw)
    sin_yaw = np.sin(half_yaw)
    cos_pitch = np.cos(half_pitch)
    sin_pitch = np.sin(half_pitch)
    cos_roll = np.cos(half_roll)
    sin_roll = np.sin(half_roll)
    ccc = cos_roll*cos_pitch*cos_yaw
    ccs = cos_roll*cos_pitch*sin_yaw
    csc = cos_roll*sin_pitch*cos_yaw
    css = cos_roll*sin_pitch*sin_yaw
    scs = sin_roll*cos_pitch*sin_yaw
    scc = sin_roll*cos_pitch*cos_yaw
    ssc = sin_roll*sin_pitch*cos_yaw
    sss = sin_roll*sin_pitch*sin_yaw

    jacobian_quat_euler = 0.5*np.array([[ssc - ccs, scs - csc, css - scc],
                               [-(csc + scs), -(ssc + ccs), ccc + sss],
                               [scc - css, ccc - sss, ccs - ssc],
                               [ccc + sss, -(css + scc), -(csc + scs)]])


    jacobian = np.block([[jacobian_quat_euler, np.zeros((4,3))],
                         [np.zeros((3,3)), np.eye(3)]
                         ])

    return jacobian

''' J_e->q -> Jacobian of convertion from Euler+3D to Quaternion+3D '''
def computeJacobian_eulerToQuat_array(q_poseEuler_array):
    half_yaw = 0.5*q_poseEuler_array[:,0]
    half_pitch = 0.5*q_poseEuler_array[:,1]
    half_roll = 0.5*q_poseEuler_array[:,2]
    cos_yaw = np.cos(half_yaw)
    sin_yaw = np.sin(half_yaw)
    cos_pitch = np.cos(half_pitch)
    sin_pitch = np.sin(half_pitch)
    cos_roll = np.cos(half_roll)
    sin_roll = np.sin(half_roll)
    ccc = cos_roll*cos_pitch*cos_yaw
    ccs = cos_roll*cos_pitch*sin_yaw
    csc = cos_roll*sin_pitch*cos_yaw
    css = cos_roll*sin_pitch*sin_yaw
    scs = sin_roll*cos_pitch*sin_yaw
    scc = sin_roll*cos_pitch*cos_yaw
    ssc = sin_roll*sin_pitch*cos_yaw
    sss = sin_roll*sin_pitch*sin_yaw

    a11 = ssc - ccs
    a12 = scs - csc
    a13 = css - scc
    a21 = -(csc + scs)
    a22 = -(ssc + ccs)
    a23 = ccc + sss
    a32 = ccc - sss
    a42 = -(css + scc)

    jacobian_array = np.zeros((q_poseEuler_array.shape[0],7,6))
    jacobian_array[:, 0, 0] = 0.5*a11
    jacobian_array[:, 0, 1] = 0.5*a12
    jacobian_array[:, 0, 2] = 0.5*a13
    jacobian_array[:, 1, 0] = 0.5*a21
    jacobian_array[:, 1, 1] = 0.5*a22
    jacobian_array[:, 1, 2] = 0.5*a23
    jacobian_array[:, 2, 0] = -0.5*a13
    jacobian_array[:, 2, 1] = 0.5*a32
    jacobian_array[:, 2, 2] = -0.5*a11
    jacobian_array[:, 3, 0] = 0.5*a23
    jacobian_array[:, 3, 1] = 0.5*a42
    jacobian_array[:, 3, 2] = 0.5*a21
    ones = np.full(q_poseEuler_array.shape[0], 1.)
    jacobian_array[:, 4, 3] = ones
    jacobian_array[:, 5, 4] = ones
    jacobian_array[:, 6, 5] = ones

    return jacobian_array

''' J_e1+e2 -> Jacobian of poses composition in Euler+3D '''
def computeJacobianEuler_composePose(q_poseEuler1, q_poseEuler2, q_poseEuler_compose_mean):
    q_poseQuat1 = fromPoseEulerToPoseQuat(q_poseEuler1)
    q_poseQuat2 = fromPoseEulerToPoseQuat(q_poseEuler2)
    q_poseQuat_compose = fromPoseEulerToPoseQuat(q_poseEuler_compose_mean)

    jacobian_quat_q1, jacobian_quat_q2 = poses_quat.computeJacobianQuat_composePose(q_poseQuat1, q_poseQuat2)

    jacobian_quatToEuler_q_compose = computeJacobian_quatToEuler(q_poseQuat_compose)
    jacobian_eulerToQuat_q1 = computeJacobian_eulerToQuat(q_poseEuler1)
    jacobian_eulerToQuat_q2 = computeJacobian_eulerToQuat(q_poseEuler2)

    jacobian_q1 = np.matmul(jacobian_quatToEuler_q_compose, np.matmul(jacobian_quat_q1, jacobian_eulerToQuat_q1))
    jacobian_q2 = np.matmul(jacobian_quatToEuler_q_compose, np.matmul(jacobian_quat_q2, jacobian_eulerToQuat_q2))

    return jacobian_q1, jacobian_q2

''' J_e1+e2 -> Jacobian of poses composition in Euler+3D '''
def computeJacobianEuler_composePose_array(q_poseEuler1, q_poseEuler2_array, q_poseEuler_compose_mean_array):
    q_poseQuat1 = fromPoseEulerToPoseQuat(q_poseEuler1)
    q_poseQuat2_array = fromPoseEulerToPoseQuat_array(q_poseEuler2_array)
    q_poseQuat_compose_array = fromPoseEulerToPoseQuat_array(q_poseEuler_compose_mean_array)

    jacobian_quat_q1, jacobian_quat_q2 = poses_quat.computeJacobianQuat_composePose_array(q_poseQuat1, q_poseQuat2_array)

    jacobian_quatToEuler_q_compose = computeJacobian_quatToEuler_array(q_poseQuat_compose_array)
    jacobian_eulerToQuat_q1 = computeJacobian_eulerToQuat(q_poseEuler1)
    jacobian_eulerToQuat_q2 = computeJacobian_eulerToQuat_array(q_poseEuler2_array)

    return np.einsum('kij,kjl->kil',jacobian_quatToEuler_q_compose, np.einsum('kij,jl->kil',jacobian_quat_q1, jacobian_eulerToQuat_q1)), \
           np.einsum('kij,kjl->kil',jacobian_quatToEuler_q_compose, np.einsum('kij,kjl->kil',jacobian_quat_q2, jacobian_eulerToQuat_q2))

def computeJacobianEuler_composePosePDFPoint_pose(q_mean, point_mean):
    cos_yaw = np.cos(q_mean[0])
    sin_yaw = np.sin(q_mean[0])
    cos_pitch = np.cos(q_mean[1])
    sin_pitch = np.sin(q_mean[1])
    cos_roll = np.cos(q_mean[2])
    sin_roll = np.sin(q_mean[2])

    a11 = -point_mean[0]*sin_yaw*cos_pitch - point_mean[1]*(sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll) + point_mean[2]*(-sin_yaw*sin_pitch*cos_roll + cos_yaw*sin_roll)
    a12 = -point_mean[0]*cos_yaw*sin_pitch + point_mean[1]*cos_yaw*cos_pitch*sin_roll + point_mean[2]*cos_yaw*cos_pitch*cos_roll
    a13 = point_mean[1]*(cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll) + point_mean[2]*(-cos_yaw*sin_pitch*sin_roll + sin_yaw*cos_roll)
    a21 = point_mean[0]*cos_yaw*cos_pitch + point_mean[1]*(cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll) + point_mean[2]*(cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll)
    a22 = -point_mean[0]*sin_yaw*sin_pitch + point_mean[1]*sin_yaw*cos_pitch*sin_roll + point_mean[2]*sin_yaw*cos_pitch*cos_roll
    a23 = point_mean[1]*(sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll) - point_mean[2]*(sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll)
    a31 = 0.
    a32 = -(point_mean[0]*cos_pitch + point_mean[1]*sin_pitch*sin_roll + point_mean[2]*sin_pitch*cos_roll)
    a33 = point_mean[1]*cos_pitch*cos_roll - point_mean[2]*cos_pitch*sin_roll

    return np.block([np.array([[a11,a12,a13],
                     [a21,a22,a23],
                     [a31,a32,a33]]),np.eye(3)])

def computeJacobianEuler_composePosePDFPoint_pose_array(q_mean, point_mean_array):
    cos_yaw = np.cos(q_mean[0])
    sin_yaw = np.sin(q_mean[0])
    cos_pitch = np.cos(q_mean[1])
    sin_pitch = np.sin(q_mean[1])
    cos_roll = np.cos(q_mean[2])
    sin_roll = np.sin(q_mean[2])

    jacobian = np.zeros((point_mean_array.shape[0], 3, 6))

    jacobian[:,0,0] = -point_mean_array[:,0]*sin_yaw*cos_pitch - point_mean_array[:,1]*(sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll) + point_mean_array[:,2]*(-sin_yaw*sin_pitch*cos_roll + cos_yaw*sin_roll)
    jacobian[:,0,1] = -point_mean_array[:,0]*cos_yaw*sin_pitch + point_mean_array[:,1]*cos_yaw*cos_pitch*sin_roll + point_mean_array[:,2]*cos_yaw*cos_pitch*cos_roll
    jacobian[:,0,2] = point_mean_array[:,1]*(cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll) + point_mean_array[:,2]*(-cos_yaw*sin_pitch*sin_roll + sin_yaw*cos_roll)
    jacobian[:,1,0] = point_mean_array[:,0]*cos_yaw*cos_pitch + point_mean_array[:,1]*(cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll) + point_mean_array[:,2]*(cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll)
    jacobian[:,1,1] = -point_mean_array[:,0]*sin_yaw*sin_pitch + point_mean_array[:,1]*sin_yaw*cos_pitch*sin_roll + point_mean_array[:,2]*sin_yaw*cos_pitch*cos_roll
    jacobian[:,1,2] = point_mean_array[:,1]*(sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll) - point_mean_array[:,2]*(sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll)
    jacobian[:,2,0] = 0.
    jacobian[:,2,1] = -(point_mean_array[:,0]*cos_pitch + point_mean_array[:,1]*sin_pitch*sin_roll + point_mean_array[:,2]*sin_pitch*cos_roll)
    jacobian[:,2,2] = point_mean_array[:,1]*cos_pitch*cos_roll - point_mean_array[:,2]*cos_pitch*sin_roll
    ones = np.full(point_mean_array.shape[0],1.)
    jacobian[:,0,3] = ones
    jacobian[:,1,4] = ones
    jacobian[:,2,5] = ones

    return jacobian

''' e1+e2 in Euler+3D PDF (with uncertainty) '''
def composePosePDFEuler(q_posePDFEuler1, q_posePDFEuler2):
    q_poseEuler_compose_mean = composePoseEuler(q_posePDFEuler1['pose_mean'], q_posePDFEuler2['pose_mean'])

    jacobian_q1, jacobian_q2 = computeJacobianEuler_composePose(q_posePDFEuler1['pose_mean'], q_posePDFEuler2['pose_mean'], q_poseEuler_compose_mean)

    q_poseEuler_compose_cov = np.matmul(jacobian_q1, np.matmul(q_posePDFEuler1['pose_cov'], np.transpose(jacobian_q1))) +\
          np.matmul(jacobian_q2, np.matmul(q_posePDFEuler2['pose_cov'], np.transpose(jacobian_q2)))

    q_posePDFEuler_compose = {'pose_mean' : q_poseEuler_compose_mean, 'pose_cov' : q_poseEuler_compose_cov}
    return q_posePDFEuler_compose

''' e1+e2 in Euler+3D PDF (with uncertainty) '''
def composePosePDFEuler_array(q_posePDFEuler1, q_posePDFEuler2_array):
    q_poseEuler_compose_mean = composePoseEuler_array(q_posePDFEuler1['pose_mean'], q_posePDFEuler2_array['pose_mean'])

    jacobian_q1, jacobian_q2 = computeJacobianEuler_composePose_array(q_posePDFEuler1['pose_mean'], q_posePDFEuler2_array['pose_mean'], q_poseEuler_compose_mean)

    q_poseEuler_compose_cov =  np.einsum('kij,kjl->kil', jacobian_q1, np.einsum('ij,klj->kil',q_posePDFEuler1['pose_cov'],jacobian_q1,optimize=True),optimize=True) + \
                               np.einsum('kij,kjl->kil', jacobian_q2,
                                         np.einsum('kij,klj->kil', q_posePDFEuler2_array['pose_cov'], jacobian_q2,
                                                   optimize=True), optimize=True)

    q_posePDFEuler_compose = {'pose_mean' : q_poseEuler_compose_mean, 'pose_cov' : q_poseEuler_compose_cov}
    return q_posePDFEuler_compose

def composePosePDFEulerPoint(q_posePDFEuler, point):
    jacobian_pose = computeJacobianEuler_composePosePDFPoint_pose(q_posePDFEuler['pose_mean'], point['mean'])
    jacobian_point = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posePDFEuler['pose_mean'][0:3]).as_dcm()

    cov = jacobian_pose@q_posePDFEuler['pose_cov']@jacobian_pose.T + jacobian_point@point['cov']@jacobian_point.T

    mean = composePoseEulerPoint(q_posePDFEuler['pose_mean'], point['mean'])
    return {'mean': mean, 'cov': cov}

def composePosePDFEulerPoint_array(q_posePDFEuler, point_array):
    jacobian_pose_array = computeJacobianEuler_composePosePDFPoint_pose_array(q_posePDFEuler['pose_mean'], point_array['mean'])
    jacobian_point = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posePDFEuler['pose_mean'][0:3]).as_dcm()

    cov = np.einsum('kij,kjl->kil', jacobian_pose_array, np.einsum('ij,klj->kil',q_posePDFEuler['pose_cov'],jacobian_pose_array,optimize=True),optimize=True) + \
          np.einsum('ij,kjl->kil', jacobian_point,
                    np.einsum('kij,lj->kil', point_array['cov'], jacobian_point,
                              optimize=True), optimize=True)

    mean = composePoseEulerPoint(q_posePDFEuler['pose_mean'], point_array['mean'])
    return {'mean': mean, 'cov': cov}

def inversePosePDFEuler(q_posePDFEuler):
    q_posePDFQuat = fromPosePDFEulerToPosePDFQuat(q_posePDFEuler)
    q_posePDFQuat_inv = poses_quat.inversePosePDFQuat(q_posePDFQuat)
    return fromPosePDFQuatToPosePDFEuler(q_posePDFQuat_inv)

''' q1 - q2 with uncertainty '''
def inverseComposePosePDFEuler(q_posePDFEuler1, q_posePDFEuler2):
    q_posePDFQuat1 = fromPosePDFEulerToPosePDFQuat(q_posePDFEuler1)
    q_posePDFQuat2 = fromPosePDFEulerToPosePDFQuat(q_posePDFEuler2)
    q_posePDFQuat_composed = poses_quat.inverseComposePosePDFQuat(q_posePDFQuat1, q_posePDFQuat2)
    return fromPosePDFQuatToPosePDFEuler(q_posePDFQuat_composed)

''' q1 - q2 with uncertainty '''
def inverseComposePosePDFEuler_array(q_posePDFEuler1_array, q_posePDFEuler2):
    q_posePDFQuat1_array = fromPosePDFEulerToPosePDFQuat_array(q_posePDFEuler1_array)
    q_posePDFQuat2 = fromPosePDFEulerToPosePDFQuat(q_posePDFEuler2)
    q_posePDFQuat_composed_array = poses_quat.inverseComposePosePDFQuat_array(q_posePDFQuat1_array, q_posePDFQuat2)
    return fromPosePDFQuatToPosePDFEuler_array(q_posePDFQuat_composed_array)

def inverseComposePosePDFEulerPoint_array(q_posePDFEuler, x_array):
    q_pdf_inv = inversePosePDFEuler(q_posePDFEuler)
    return composePosePDFEulerPoint_array(q_pdf_inv, x_array)

''' Convertion from Euler+3D PDF to Quaternion+3D PDF '''
def fromPosePDFEulerToPosePDFQuat(q_posePDFEuler):
    q_posePDFQuat_mean = fromPoseEulerToPoseQuat(q_posePDFEuler['pose_mean'])
    J = computeJacobian_eulerToQuat(q_posePDFEuler['pose_mean'])
    q_posePDFQuat_cov = np.matmul(J, np.matmul(q_posePDFEuler['pose_cov'], np.transpose(J)))
    return {'pose_mean' : q_posePDFQuat_mean, 'pose_cov': q_posePDFQuat_cov}

''' Convertion from Euler+3D PDF to Quaternion+3D PDF '''
def fromPosePDFEulerToPosePDFQuat_array(q_posePDFEuler_array):
    q_posePDFQuat_mean_array = fromPoseEulerToPoseQuat_array(q_posePDFEuler_array['pose_mean'])
    J = computeJacobian_eulerToQuat_array(q_posePDFEuler_array['pose_mean'])
    q_posePDFQuat_cov_array = np.einsum('kij,kjl->kil', J, np.einsum('kij,klj->kil',q_posePDFEuler_array['pose_cov'],J,optimize=True),optimize=True)  #np.matmul(J, np.matmul(q_posePDFEuler['pose_cov'], np.transpose(J)))
    return {'pose_mean' : q_posePDFQuat_mean_array, 'pose_cov': q_posePDFQuat_cov_array}

''' Convertion from Quaternion+3D PDF to Euler+3D PDF '''
def fromPosePDFQuatToPosePDFEuler(q_posePDFQuat):
    q_posePDFEuler_mean = fromPoseQuatToPoseEuler(q_posePDFQuat['pose_mean'])
    J = computeJacobian_quatToEuler(q_posePDFQuat['pose_mean'])
    q_posePDFEuler_cov = np.matmul(J, np.matmul(q_posePDFQuat['pose_cov'], np.transpose(J)))
    return {'pose_mean':q_posePDFEuler_mean, 'pose_cov':q_posePDFEuler_cov}

''' Convertion from Quaternion+3D PDF to Euler+3D PDF '''
def fromPosePDFQuatToPosePDFEuler_array(q_posePDFQuat_array):
    q_posePDFEuler_mean = fromPoseQuatToPoseEuler_array(q_posePDFQuat_array['pose_mean'])
    J = computeJacobian_quatToEuler_array(q_posePDFQuat_array['pose_mean'])
    q_posePDFEuler_cov = np.einsum('kij,kjl->kil', J, np.einsum('kij,klj->kil',q_posePDFQuat_array['pose_cov'],J,optimize=True),optimize=True)
    return {'pose_mean':q_posePDFEuler_mean, 'pose_cov':q_posePDFEuler_cov}

def distanceSE3(p1, p2):
    return math_utility.distanceSE3(p1, p2)
import numpy as np
import scipy as scp
import math_utility
import gaussianICP
import poses_euler
import poses_quat
import plotPICP
import dataAssociation
import time

import gaussianPICP
from dataPICP import *

''' TODO : refactor with classes and inheritence (also for poses euler and quat !)'''

def generatePointClouds(q_gt_mean, n, cov_scale):
    c_pts_array = {"mean": None, "cov": None}
    a_pts_array = {"mean": None, "cov": None}

    ''' Generate means '''
    points = []
    d = 10
    for i in range(0, n):
        point = [np.random.uniform(-d, d), np.random.uniform(-d, d), np.random.uniform(-d, d)]
        points.append(point)
    c_pts_array["mean"] = np.array(points)
    ''' Generate covariance '''
    c_pts_array["cov"] = np.empty((n, 3, 3))
    for i in range(0, n):
        mat = cov_scale*np.random.rand(3,3)
        c_pts_array["cov"][i] = mat.T@mat

    ''' Sample from c_pts_array the ground truth point '''
    c_gt_points = np.empty((n, 3))
    i = 0
    for mean_c, cov_c in zip(c_pts_array["mean"], c_pts_array["cov"]):
        c_gt_points[i] = np.random.multivariate_normal(mean_c, cov_c)
        i += 1

    ''' Compute the second cloud from the ground truth point of the first cloud '''
    rot = scp.spatial.transform.Rotation.from_euler('ZYX', q_gt_mean[0:3])
    a_samples = rot.apply(c_gt_points) + q_gt_mean[3:6]

    ''' Generate the second point cloud distributions '''
    a_pts_array["cov"] = np.empty((n, 3, 3))
    a_pts_array["mean"] = np.empty((n, 3))
    for i in range(0, n):
        mat = cov_scale * np.random.rand(3, 3)
        a_pts_array["cov"][i] = mat.T @ mat
        a_pts_array["mean"][i] = np.random.multivariate_normal(a_samples[i], a_pts_array["cov"][i])

    return a_pts_array, c_pts_array

def experiment(q_gt_mean, q0, maxIter, picp):
    q, Fval, path = picp.LevenbergMarquardt(q0, maxIter)

    ''' Compute distances at each step to the ground truth transformation '''
    distancesSE3 = []
    for pose in path['pose']:
        ''' Ugly. Will be changed when refactoring with classes '''
        if (q_gt_mean.shape[0] == 6):
            distancesSE3.append(poses_euler.distanceSE3(pose['pose_mean'], q_gt_mean))
        else:
            distancesSE3.append(poses_quat.distanceSE3(pose['pose_mean'], q_gt_mean))
    return dataPICP(q, path, distancesSE3)

def experiment_expMax(q_gt_mean, q0, maxIter, picp):
    q, Fval, path = picp.LevenbergMarquardt_expMax(q0, maxIter)

    ''' Compute distances at each step to the ground truth transformation '''
    distancesSE3 = []
    for pose in path['pose']:
        ''' Ugly. Will be changed when refactoring with classes '''
        if (q_gt_mean.shape[0] == 6):
            distancesSE3.append(poses_euler.distanceSE3(pose['pose_mean'], q_gt_mean))
        else:
            distancesSE3.append(poses_quat.distanceSE3(pose['pose_mean'], q_gt_mean))
    return dataPICP(q, path, distancesSE3)

def experiment_compare_oneTrial(nExp, n, cov_scale_point, cov_scale_q, maxIter):
    ''' For numerical jacobian in Euler and Quaternion cases '''
    increments_euler = np.full((6,), 1e-5)
    increments_quat = np.full((7,), 1e-5)

    ''' Generate the simulated data '''
    mat = cov_scale_q*np.random.rand(6, 6)
    q_euler_cov = mat.T @ mat
    q0_euler_mean = np.random.rand(6)
    q0_euler = {"pose_mean": q0_euler_mean, "pose_cov": q_euler_cov}
    q0_quat = poses_euler.fromPosePDFEulerToPosePDFQuat(q0_euler)

    ''' Sample initial transformation '''
    q_gt_euler_mean = np.random.multivariate_normal(q0_euler_mean, q_euler_cov)
    q_gt_quat_mean = poses_euler.fromPoseEulerToPoseQuat(q_gt_euler_mean)

    a_pts_array, c_pts_array = generatePointClouds(q_gt_euler_mean, n, cov_scale_point)
    #C_x = gaussianICP.compute_C_x(c_pts_array)
   # kargsInit = {"a_pts_array": a_pts_array, "c_pts_array": c_pts_array, "C_x": C_x}

    ''' Distance initiale '''
    d0 = poses_euler.distanceSE3(q0_euler["pose_mean"], q_gt_euler_mean)

    datas = []
    labels = []

    '''picp_euler = gaussianPICP.gaussianPICP_Euler(a_pts_array, c_pts_array)
    dataEuler = experiment(q_gt_euler_mean,
                           q0_euler,
                           maxIter,
                           picp_euler)
    data.append(dataEuler)
    labels.append("Euler")


    picp_quat = gaussianPICP.gaussianPICP_Quaternion(a_pts_array, c_pts_array)
    dataQuaternion = experiment(q_gt_quat_mean,
                                q0_quat,
                                maxIter,
                                picp_quat)

    data.append(dataQuaternion)
    labels.append("Quaternion")'''


    picp_se = gaussianPICP.gaussianPICP_se(a_pts_array, c_pts_array)
    data_se = experiment(q_gt_euler_mean,
                         q0_euler,
                         maxIter,
                         picp_se)
    datas.append(data_se)
    labels.append("SE")

    picp_se_expMax = gaussianPICP.gaussianPICP_se(a_pts_array, c_pts_array)
    data_se_expMax = experiment_expMax(q_gt_euler_mean,
                         data_se.q_opt,
                         maxIter,
                         picp_se_expMax)
    datas.append(data_se_expMax)
    labels.append("SE_expMax")

    ''' Display and Save results '''
    colors = ['r','g','b']
    plotPICP.plotGraphs_DistanceWRTIteration(0, datas, colors, labels,linestyles=['-','-'], linewidth=1, alpha=0.6)
    plt.show()

    return d0, distances[0][-1], distances[1][-1], distances[2][-1], paths[0]["cost"][-1], paths[1]["cost"][-1], paths[2]["cost"][-1]

def experiment_compare(nTrial, n, cov_scale_point, cov_scale_q, maxIter):
    d_quat_relatif = []
    d_euler_relatif = []
    d_se_relatif = []
    for i in range(0, nTrial):
        print("Experiment i : {}".format(i))
        np.random.seed(i)
        d0, d_euler, d_quat, d_se, cost_euler, cost_quat, cost_se = experiment_compare(i, n, cov_scale_point, cov_scale_q, maxIter)
        d_quat_relatif.append(d0 / d_quat)
        d_euler_relatif.append(d0 / d_euler)
        d_se_relatif.append(d0 / d_se)

    folder = "experiment_gaussianPICP_compare/"

    with open(folder + "experiment_data_PICP_euler.txt","w") as file:
        for d in d_euler_relatif:
            file.write(str(d) + "\n")
    with open(folder + "experiment_data_PICP_quat.txt","w") as file:
        for d in d_quat_relatif:
            file.write(str(d) + "\n")
    with open(folder + "experiment_data_PICP_se.txt","w") as file:
        for d in d_se_relatif:
            file.write(str(d) + "\n")

# ------------- MAIN ---------------------
if __name__ == "__main__":
    #np.random.seed(6)
    #d0, d_euler, d_quat, d_se, cost_euler, cost_quat, cost_se = experiment_compare_oneTrial(6, 100, 0.5, 0.5, 100)

    '''experiment_compoare(500, 100, 0.25, 0.1, 100)

    d_euler_relatif = np.genfromtxt("experiment_data_PICP_euler.txt")
    d_quat_relatif = np.genfromtxt("experiment_data_PICP_quat.txt",)
    d_se_relatif = np.genfromtxt("experiment_data_PICP_se.txt")'''

    plotPICP.plotBoxPlot_comparisonRepresentation_("experiment_gaussianPICP_compare/experiment_data_euler.txt",
                                                  "experiment_gaussianPICP_compare/experiment_data_quat.txt",
                                                  "experiment_gaussianPICP_compare/experiment_data_se.txt",
                                                  True,
                                                  "comparisonRepresentation")
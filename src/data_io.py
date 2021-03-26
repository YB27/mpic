import numpy as np
import scipy
import os

def writePoseFunc(file, q):
    file.write(str(q[0]) + "," + str(q[1]) + "," + str(q[2]) + "," +
               str(q[3]) + "," + str(q[4]) + "," + str(q[5]) + "\n")

def getSubFoldersSortedList(input_folder):
    folders_int = [int(x) for x in os.listdir(input_folder) if
                   os.path.isdir(os.path.join(input_folder, x))]  # [int(x) for x in os.listdir(input_folder)]
    folders_sorted = [str(x) for x in sorted(folders_int)]
    return folders_sorted

def loadDistancesInFolder_singleFile(file):
    distances = [[], [], [], [], []]
    with(open(file, "r")) as file:
        file.readline() #header
        for line in file:
            line_parsed = line.split(',')
            for i in range(0, len(distances)):
                 distances[i].append(float(line_parsed[i]))
    return distances

def loadDistancesPosesFromFile(fileName):
    distances = []
    poses = []
    with open(fileName, "r") as file:
        file.readline()  # skip header
        for line in file:
            line_parsed = line.split(",")
            distances.append(float(line_parsed[0]))
            pose = []
            for i in range(1,7):
                pose.append(float(line_parsed[i]))
            poses.append(pose)
    return distances, poses

def loadDistancesInFolder(folder, suffix):
    suffixTxt = suffix + ".txt"
    files = ["dist_init" + suffixTxt, "dist2D" + suffixTxt, "dist3D" + suffixTxt, "dist3D_allArcs"+ suffixTxt,
             #"dist3D_plane"+ suffixTxt,
             "dist3D_plane_startPoint"+ suffixTxt]
    distances = [[], [], [], [], []]
    i = 0
    for f in files:
        distances[i],_ = loadDistancesPosesFromFile(folder + "/" + f)
        i+=1

    return distances

def readNormal(file):
    line = file.readline()
    line_parsed = line.split(',')
    normal_mean = np.array(
        [float(line_parsed[0]), float(line_parsed[1]), float(line_parsed[2])])
    normal_cov = []
    for j in range(3, 12):
        normal_cov.append(float(line_parsed[j]))
    normal_cov = np.array(normal_cov).reshape(3, 3)
    return normal_mean, normal_cov

def loadNormalsFromFile(estimatedNormalsFile, nonUniformIdxs):
    nFirstScanSize = len(nonUniformIdxs[0])
    nSecondScanSize = len(nonUniformIdxs[1])
    normals_firstScanFrame = {"mean": np.empty((nFirstScanSize, 3)), "cov": np.empty((nFirstScanSize, 3, 3))}
    normals_secondScanFrame = {"mean": np.empty((nSecondScanSize, 3)), "cov": np.empty((nSecondScanSize, 3, 3))}
    with open(estimatedNormalsFile, "r") as file:
        file.readline()  # skip the headers
        for i in range(0, nFirstScanSize):
            normal_mean, normal_cov = readNormal(file)
            normals_firstScanFrame["mean"][i, :] = normal_mean
            normals_firstScanFrame["cov"][i, :, :] = normal_cov

        for i in range(0, nSecondScanSize):
            normal_mean, normal_cov = readNormal(file)
            normals_secondScanFrame["mean"][i, :] = normal_mean
            normals_secondScanFrame["cov"][i, :, :] = normal_cov

    return normals_firstScanFrame, normals_secondScanFrame

def normalsInScanReferenceFrame(normals_inFirstFrame, scanRefFrame):
    rot_inv = scipy.spatial.transform.Rotation.from_euler('ZYX', scanRefFrame['pose_mean'][0:3]).inv()
    R = rot_inv.as_matrix()
    normals_inFirstFrame["mean"] = rot_inv.apply(np.array(normals_inFirstFrame["mean"]))
    normals_inFirstFrame["cov"] = np.einsum("ij,kjl->kil", R,
                                              np.einsum("kij,lj->kil", normals_inFirstFrame["cov"], R))

def loadNormals(estimatedNormalsFile, firstScan_refFrame, secondScan_refFrame, nonUniformIdxs):
    normals_firstScanFrame, normals_secondScanFrame = loadNormalsFromFile(estimatedNormalsFile, nonUniformIdxs)

    ''' Express the normals in the first/second scan reference frame '''
    normalsInScanReferenceFrame(normals_firstScanFrame, firstScan_refFrame)
    normalsInScanReferenceFrame(normals_secondScanFrame, secondScan_refFrame)

    return normals_firstScanFrame, normals_secondScanFrame

def readPoseMean(line_parsed):
    poseMean = np.empty((6,))
    poseMean[0] = float(line_parsed[4])
    poseMean[1] = float(line_parsed[5])
    poseMean[2] = float(line_parsed[6])
    poseMean[3] = float(line_parsed[1])
    poseMean[4] = float(line_parsed[2])
    poseMean[5] = float(line_parsed[3])
    return poseMean

def readPoseCov(line_parsed):
    cov = []
    for i in range(7, 43):
        cov.append(float(line_parsed[i]))
    poseCov = np.array(cov).reshape(6, 6)
    xyzCov = poseCov[0:3, 0:3].copy()
    off_block = poseCov[0:3, 3:6].copy()
    poseCov[0:3, 0:3] = poseCov[3:6, 3:6]
    poseCov[3:6, 3:6] = xyzCov
    poseCov[0:3, 3:6] = poseCov[3:6, 0:3]
    poseCov[3:6, 0:3] = off_block
    return poseCov

def readPosePDF(line):
    line_parsed = line.split(',')

    timeStamp = int(line_parsed[0])
    pose = {'pose_mean': readPoseMean(line_parsed), 'pose_cov': readPoseCov(line_parsed)}

    return timeStamp, pose

def loadTrajectoryFromFile(fileName):
    trajectory = {}
    with open(fileName) as file:
        file.readline()  # skip the headers
        for line in file:
            timeStamp, pose = readPosePDF(line)
            trajectory[timeStamp] = pose

    return trajectory

def loadHorizontalSonarSimulatedData(horizontalSonarDataFile, firstScan_poses_relative, secondScan_poses_relative, rho_std, psi_std, b):
    scanMeas = [[] for i in range(2)]
    nonUniformIdxs = [[],[]]
    idx = 0
    with open(horizontalSonarDataFile, "r") as file:
        file.readline()  # skip the headers
        for line in file:
            line_parsed = line.split(',')
            scan_idx = int(line_parsed[0])
            poseInScanIdx = int(line_parsed[5])
            arcIdx = int(line_parsed[6])
            if (scan_idx == 0):
                pose = firstScan_poses_relative[poseInScanIdx]
            else:
                pose = secondScan_poses_relative[poseInScanIdx]

            alpha = float(line_parsed[3])
            beta = float(line_parsed[4])
            scanMeas[scan_idx].append({'rho': {'mean': float(line_parsed[1]), 'std': rho_std},
                                        'theta': {'alpha': alpha, 'beta': beta,
                                                  'b': b},
                                        'psi': {'mean': float(line_parsed[2]), 'std': psi_std},
                                        'pose_mean': pose['pose_mean'], 'pose_cov': pose['pose_cov'],
                                        'arcIdx': arcIdx})
            if(alpha > 1 and beta > 1):
                nonUniformIdxs[scan_idx].append(idx)
            idx += 1

    return scanMeas[0], scanMeas[1], nonUniformIdxs
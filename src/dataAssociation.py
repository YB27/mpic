import numpy as np
import scipy.stats.distributions

def mahalanobisMetric(points1, points2):
    n = points1['mean'].shape[1]
    nPoints1 = points1['mean'].shape[0]
    nPoints2 = points2['mean'].shape[0]

    smallCov = 1e-6*np.eye(n)
    for i in range(0,nPoints1):
        points1['cov'][i,:,:] += smallCov
    for i in range(0, nPoints2):
        points2['cov'][i, :, :] += smallCov

    distances = np.empty((nPoints1, nPoints2, n))
    meanCovs_inv = np.empty((nPoints1, nPoints2, n, n))
    for i in range(0, nPoints1):
        distances[i,:,:] = (points2['mean'] - points1['mean'][i])
        meanCovs_inv[i,:,:,:] = 0.5*(np.linalg.inv(points2['cov']) + np.linalg.inv(points1['cov'][i]))

    res = np.einsum('ijk,ijk->ij', distances, np.einsum('ijkl,ijl->ijk', meanCovs_inv, distances))
    return np.sqrt(res)

''' TODO : Implement a look-up approach on octree as described in 
Albert Palomer, Pere Ridao, and David Ribas. “Multibeam 3D underwater SLAM with probabilis-
tic registration”. In: Sensors 16.4 (2016) '''
def dataAssociation_withDistanceMatrix(distancesMatrix, threshold):
    idxs_pointCloud_1 = []
    idxs_pointCloud_2 = []
    indiv_compatible_A = [set() for i in range(distancesMatrix.shape[0])]

    ''' Sort per distance in increasing order '''
    ''' https://stackoverflow.com/questions/29734660/python-numpy-keep-a-list-of-indices-of-a-sorted-2d-array '''
    orderedIndexes = np.vstack(np.unravel_index(np.argsort(distancesMatrix, axis=None, kind='mergesort'), distancesMatrix.shape)).T

    alreadyAssociatedPoint2 = set()
    alreadyAssociatedPoint1 = set()
    for assocIndexes in orderedIndexes:
        compatible = (distancesMatrix[assocIndexes[0], assocIndexes[1]] < threshold)
        if(compatible):
             indiv_compatible_A[assocIndexes[0]].add(assocIndexes[1])

        if (assocIndexes[1] not in alreadyAssociatedPoint2 and
            assocIndexes[0] not in alreadyAssociatedPoint1 and
            compatible):
            #associations[assocIndexes[0]] = assocIndexes[1]
            idxs_pointCloud_1.append(assocIndexes[0])
            idxs_pointCloud_2.append(assocIndexes[1])
            alreadyAssociatedPoint1.add(assocIndexes[0])
            alreadyAssociatedPoint2.add(assocIndexes[1])

    return np.array(idxs_pointCloud_1), np.array(idxs_pointCloud_2), indiv_compatible_A #associations

''' Associate point from a point cloud to another '''
''' Here, points are probabilistic and depending on it, we use different metric '''
''' Return a vector of association assoc[point1_idx] = point2_idx or -1 if no valid association was found '''
def dataAssociation(func_metric, pointCloud_1, pointCloud_2, threshold, **kwargs):
    ''' Return the distances (matrix nPoint1 X nPoint2) '''
    if(len(kwargs) == 0):
        distances = func_metric(pointCloud_1, pointCloud_2)
    else:
        distances = func_metric(**kwargs)

    return dataAssociation_withDistanceMatrix(distances, threshold)

def testMahalanobisMetric():
    A1 = np.random.rand(3,3) - 0.5
    A2 = np.random.rand(3,3) - 0.5
    cov1 = A1.T@A1
    cov2 = A2.T@A2
    points1 = {'mean': np.array([[1.,2.,3.],
                                 [4.,5.,6.],
                                 [4.,5.,6.]]), 'cov': [cov1, cov2, cov1]}

    points2 = {'mean': np.array([[6.,5.,4.],
                                 [3.,2.,1.]]), 'cov': [cov2, cov1]}

    for i in range(0,points1['mean'].shape[0]):
        for j in range(0,points2['mean'].shape[0]):
            print("indexes : {}".format(str(i) + ',' + str(j)))
            diff = points1['mean'][i] - points2['mean'][j]
            meanCov = 0.5*(np.linalg.inv(points1['cov'][i]) + np.linalg.inv(points2['cov'][j]))
            d_mahala = np.dot(diff, meanCov@diff)
            print("d_mahala : {}".format(np.sqrt(d_mahala)))

    print("Result with function mahalanobisMetric :")
    print(mahalanobisMetric(points1, points2))

def testDataAssoc():
    points1 = {'mean': np.array([[1., 1., 1.],
                                 [2., 2., 2.],
                                 [3,3,3]]), 'cov': [0.4 * np.eye(3),0.4  * np.eye(3),0.4  * np.eye(3)]}

    points2 = {'mean': np.array([[3.1,3.1,3.1],
                                 [1.1, 0.9, 1.1],
                                 [2.1,1.9,1.9]]), 'cov': [0.4  * np.eye(3), 0.4*np.eye(3),0.4*np.eye(3) ]}

    threshold = scipy.stats.distributions.chi2.ppf(0.95,df=3)
    associations = dataAssociation(mahalanobisMetric, points1, points2, threshold)
    print("Assoc mahalanobis : ")
    print(associations)

# ------------- MAIN ---------------------
if __name__ == "__main__":
    testMahalanobisMetric()
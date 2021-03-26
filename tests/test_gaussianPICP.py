import numpy as np
import scipy.spatial
import time
import sonarPICP
import poses_euler
import math_utility
import gaussianPICP
import unittest

''' Function cost mahalanobis e^T Pe^-1 e '''
def functionCost_array_test(q, a_pts_array, c_pts_array, C_x, eps):
    R,t = math_utility.exp_SE3(eps)
    q_eps = np.empty((6,))
    q_eps[3:6] = t
    q_eps[0:3] = scipy.spatial.transform.Rotation.from_matrix(R).as_euler('ZYX')
    q_ = {"pose_mean": poses_euler.composePoseEuler(q["pose_mean"], q_eps),
          "pose_cov": q["pose_cov"]}
    gPICP_ = gaussianPICP.gaussianPICP_se(a_pts_array, c_pts_array)
    gPICP_.initializeData()
    gPICP_.updateForLM(q_)

    return gPICP_.costFunction(q_)

def sigma_inv(q,a_pts_array, c_pts_array, U, C_x, eps, t_):
    R, t = math_utility.exp_SE3(t_[0]*eps)
    q_eps = np.empty((6,))
    q_eps[3:6] = t
    q_eps[0:3] = scipy.spatial.transform.Rotation.from_dcm(R).as_euler('ZYX')
    q_ = {"pose_mean": poses_euler.composePoseEuler(q["pose_mean"], q_eps),
          "pose_cov": q["pose_cov"]}

    errors_mean, errors_cov_inv, omega, K, R = updateForLM(q_, a_pts_array, c_pts_array, C_x)
    return errors_cov_inv[0,0,0]

def functionCost_i(q, ai, ci, cov_ai, cov_ci, eps):
    R, t = math_utility.exp_SE3(eps)
    q_eps = np.empty((6,))
    q_eps[3:6] = t
    q_eps[0:3] = scipy.spatial.transform.Rotation.from_matrix(R).as_euler('ZYX')
    q_ = {"pose_mean": poses_euler.composePoseEuler(q["pose_mean"], q_eps),
          "pose_cov": q["pose_cov"]}

    R_ = scipy.spatial.transform.Rotation.from_euler('ZYX', q_["pose_mean"][0:3]).as_matrix()
    ci_x = math_utility.vectToSkewMatrix(ci)
    U_i = np.zeros((3,6))
    U_i[:,0:3] = -ci_x
    U_i[0,3] = 1.
    U_i[1,4] = 1.
    U_i[2,5] = 1.

    omega_i = cov_ci + U_i@q["pose_cov"]@U_i.T
    ni = poses_euler.composePoseEulerPoint(q_["pose_mean"], ci)
    ni_cov = R_@omega_i@(R_.T)
    error_cov_inv_i = np.linalg.inv(cov_ai + ni_cov)
    error_mean_i = ni - ai

    return np.dot(error_mean_i, error_cov_inv_i@error_mean_i)

def H_zi_numForTest(q, ai, ci, cov_ai, cov_ci):
    dincr = 1e-4
    eps_null = np.zeros((6,))

    H_a_i = math_utility.numericalJacobian(lambda a: math_utility.numericalJacobian(lambda x: functionCost_i(q, a, ci, cov_ai, cov_ci, x), 1, eps_null,
                                           [dincr, dincr, dincr, dincr, dincr, dincr]).T, 6, ai, [dincr, dincr, dincr])
    H_c_i = math_utility.numericalJacobian(
        lambda c: math_utility.numericalJacobian(lambda x: functionCost_i(q, ai, c, cov_ai, cov_ci,x), 1, eps_null,
                                                 [dincr, dincr, dincr, dincr, dincr, dincr]).T, 6, ci,
        [dincr, dincr, dincr])

    return H_a_i, H_c_i


def generatePoints(q, n):
    c_pts_array = {"mean": None, "cov": None}
    a_pts_array = {"mean": None, "cov": None}

    ''' Generate means '''
    points = []
    for i in range(0, n):
        # point = [1.,1.,1.]
        point = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
        points.append(point)
    c_pts_array["mean"] = np.array(points)
    rot = scipy.spatial.transform.Rotation.from_euler('ZYX', q[0:3])
    # a_pts_array["mean"] = np.array([[0.,0.,0.]])
    a_pts_array["mean"] = rot.apply(c_pts_array["mean"]) + q[3:6]

    ''' Generate covariance '''
    c_pts_array["cov"] = np.empty((n, 3, 3))
    a_pts_array["cov"] = np.empty((n, 3, 3))
    for i in range(0, n):
        mat = np.random.rand(3, 3)
        c_pts_array["cov"][i] = mat.T @ mat
        # c_pts_array["cov"][i] = np.eye(3)
        mat = np.random.rand(3, 3)
        a_pts_array["cov"][i] = mat.T @ mat
        # a_pts_array["cov"][i] = np.eye(3)
    return a_pts_array, c_pts_array

def generateTransformation():
    q_true = np.array([0.1, -0.3, 0.5, 1.2, 0.1, -0.6])
    mat = np.random.rand(6, 6)
    q = {"pose_mean": np.array([-0.15, -0.1, 0.2, 0.2, -0.1, 0.4]), "pose_cov": mat.T @ mat}
    return q_true, q

def generateGaussianPICP(q_generate, q, nPts):
    a_pts_array, c_pts_array = generatePoints(q_generate, nPts)

    gPICP = gaussianPICP.gaussianPICP_se(a_pts_array, c_pts_array)
    gPICP.initializeData()
    gPICP.updateForLM(q)

    return gPICP

class TestGaussianPICP(unittest.TestCase):

    def test_costFunctionJacobian_se(self):
        q_true, q = generateTransformation()
        gPICP = generateGaussianPICP(q_true, q, 20)

        fval = gPICP.costFunction(q)
        start = time.time()
        J_closedForm = gPICP.costFunctionJacobian(q, fval)
        print("Comp time exact : {}".format(time.time() - start))

        dincr = 1e-6
        eps_null = np.zeros((6,))
        start = time.time()
        J_num = math_utility.numericalJacobian(lambda x: functionCost_array_test(q, gPICP.a_pts_array, gPICP.c_pts_array, gPICP.C_x, x), 1, eps_null,
                                               [dincr, dincr, dincr, dincr, dincr, dincr])
        print("Comp time num : {}".format(time.time() - start))

        self.assertTrue(np.allclose(J_closedForm, J_num, rtol=1e-05, atol=1e-08))

    def test_hessianCost(self):
        '''np.random.seed(1)
        n = 10
        q_true = np.array([0.1, -0.3, 0.5, 1.2, 0.1, -0.6])

        mat = np.random.rand(6, 6)
        q = {"pose_mean": np.array([-0.15, -0.1, 0.2, 0.2, -0.1, 0.4]), "pose_cov": mat.T @ mat}
        #q = {"pose_mean": np.array([0.,0.,0., 1.,1.,1.]), "pose_cov": np.eye(6)}
        a_pts_array, c_pts_array = generatePoints(q_true, n)'''

        q_true, q = generateTransformation()
        gPICP = generateGaussianPICP(q_true, q, 20)

        gPICP.hessianMatrixCost(q)

        dincr = 1e-4
        eps_null = np.zeros((6,))
        H_q_num = math_utility.numericalHessian(lambda x: functionCost_array_test(q, gPICP.a_pts_array, gPICP.c_pts_array, gPICP.C_x, x)**2, 1, eps_null,
                                               [dincr, dincr, dincr, dincr, dincr, dincr])

        self.assertTrue(np.allclose(gPICP.H_q, H_q_num, rtol=1e-05, atol=1e-08))
        #self.assertAlmostEqual(gPICP.H_q, H_q_num, places=5)

        '''print("H_a_closedForm :")
        print(H_z_closedForm[:,:,0:3])
        print("H_c_closedForm :")
        print(H_z_closedForm[:, :, 3:6])'''

        i = 0
        for ai, ci, cov_ai, cov_ci in zip(gPICP.a_pts_array["mean"], gPICP.c_pts_array["mean"], gPICP.a_pts_array["cov"], gPICP.c_pts_array["cov"]):
            H_ai_num, H_ci_num = H_zi_numForTest(q, ai, ci, cov_ai, cov_ci)
            self.assertTrue(np.allclose(gPICP.H_z[i,:,0:3], H_ai_num, rtol=1e-05, atol=1e-05))
            self.assertTrue(np.allclose(gPICP.H_z[i, :, 3:6], H_ci_num, rtol=1e-05, atol=1e-05))
            #print("H_c_closedForm / H_ci_num:")
            #print(gPICP.H_z[i, :, 3:6])
            #print(H_ci_num)
            i+=1


    def test_LMCovariance(self):
        np.random.seed(1)
        n = 100
        q_true, q = generateTransformation()
        gPICP = generateGaussianPICP(q_true, q, n)

        dincr = 1e-4
        eps_null = np.zeros((6,))
        H_q_num = math_utility.numericalHessian(lambda x: functionCost_array_test(q, gPICP.a_pts_array, gPICP.c_pts_array, gPICP.C_x, x) ** 2,
                                                1, eps_null,
                                                [dincr, dincr, dincr, dincr, dincr, dincr])

        H_z_num = np.zeros((6,6*n))
        i=0
        for ai, ci, cov_ai, cov_ci in zip(gPICP.a_pts_array["mean"], gPICP.c_pts_array["mean"], gPICP.a_pts_array["cov"], gPICP.c_pts_array["cov"]):
            index = 6*i
            H_ai_num, H_ci_num = H_zi_numForTest(q, ai, ci, cov_ai, cov_ci)
            H_z_num[:, index: index+3] = H_ai_num
            H_z_num[:, index+3: index+6] = H_ci_num
            i+=1

        Sigma_z = np.zeros((6*n, 6*n))
        for i in range(0,n):
            index = 6*i
            Sigma_z[index:index+3, index:index+3] = gPICP.a_pts_array["cov"][i]
            Sigma_z[index+3:index + 6, index+3:index + 6] = gPICP.c_pts_array["cov"][i]

        H_q_inv_num = np.linalg.inv(H_q_num)
        cov_num = H_q_inv_num@H_z_num@Sigma_z@H_z_num.T@H_q_inv_num

        cov_closedForm = gPICP.LM_covariance(q)

        self.assertTrue(np.allclose(cov_closedForm, cov_num, rtol=1e-05, atol=1e-08))

    def projToPlane(self, firstCloudPt_mean, secondCloudPt_mean, normal_mean):
        return secondCloudPt_mean - np.dot(normal_mean,secondCloudPt_mean - firstCloudPt_mean)*normal_mean

    def test_jacobianProjectionToPlane_wrt_n(self):
        nPts = 2
        q_true, q = generateTransformation()
        gPICP = generateGaussianPICP(q_true, q, nPts)

        # Generate normals
        gPICP.normalsInFirstScan_refFrame["mean"] = np.random.rand(nPts, 3)

        jacobian_computed = gPICP.jacobianProjectionToPlane_wrt_n(gPICP.normalsInFirstScan_refFrame["mean"])
        print("Jacobian computed : ")
        print(jacobian_computed)

        dincr = 1e-6
        for i in range(0,nPts):
            jacobian_num = math_utility.numericalJacobian(lambda x: self.projToPlane(gPICP.a_pts_array["mean"][i],x ,
                                                                                gPICP.normalsInFirstScan_refFrame["mean"][i]),
                                                                                3,
                                                                                gPICP.c_pts_array["mean"][i],
                                                                                np.full((3,), dincr))
            print("Jacobian num :")
            print(jacobian_num)

    def test_jacobianProjectionToPlane_wrt_normal(self):
        nPts = 2
        q_true, q = generateTransformation()
        gPICP = generateGaussianPICP(q_true, q, nPts)

        # Generate normals
        gPICP.normalsInFirstScan_refFrame["mean"] = np.random.rand(nPts, 3)
        for i in range(0,nPts):
            gPICP.normalsInFirstScan_refFrame["mean"][i] = gPICP.normalsInFirstScan_refFrame["mean"][i] / np.linalg.norm(gPICP.normalsInFirstScan_refFrame["mean"][i])

        vecs = gPICP.c_pts_array["mean"] - gPICP.a_pts_array["mean"]
        dot = np.einsum('ij,ij->i', vecs, gPICP.normalsInFirstScan_refFrame["mean"])
        jacobian_computed = gPICP.jacobianProjectionToPlane_wrt_normal(gPICP.normalsInFirstScan_refFrame["mean"], vecs, dot)
        print("Jacobian computed : ")
        print(jacobian_computed)

        dincr = 1e-6
        for i in range(0, nPts):
            jacobian_num = math_utility.numericalJacobian(lambda x: self.projToPlane(gPICP.a_pts_array["mean"][i],
                                                                                     gPICP.c_pts_array["mean"][i],
                                                                                     x / np.linalg.norm(x)),
                                                          3,
                                                          gPICP.normalsInFirstScan_refFrame[
                                                              "mean"][i],
                                                          np.full((3,), dincr))
            print("Jacobian num :")
            print(jacobian_num)



# ------------- MAIN ---------------------
if __name__ == "__main__":
    #t = TestGaussianPICP()
    #t.test_jacobianProjectionToPlane_wrt_normal()
    unittest.main()
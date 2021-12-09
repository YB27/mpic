import numpy as np
import scipy as scp
import scipy.spatial.transform
import poses_euler
import poses_quat
import poses2D
import math_utility
import math
import time
import PICP_display
import dataAssociation
import multiprocessing as mp
from abc import ABC, abstractmethod
import betaDist

G = np.zeros((6,3,3))
G[0] = np.array([[0,0,0],
                 [0,0,-1],
                 [0,1,0]])
G[1] = np.array([[0, 0, 1],
                 [0, 0, 0],
                 [-1, 0, 0]])
G[2] = np.array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 0]])

''' Base class for 3D Gaussian PICP '''
class gaussianPICP(ABC):
    def __init__(self, a_pts_array, c_pts_array):
        self.a_pts_array = a_pts_array
        self.a_pts_array_current = None
        self.a_pts_array_new = None
        self.c_pts_array = c_pts_array
        self.a_pts_array_assoc = a_pts_array
        self.c_pts_array_assoc = c_pts_array
        self.errors_mean = None
        self.errors_cov_inv = None
        self.associationType = "pointToPoint"
        self.normalsInFirstScan_refFrame = {"mean" : None, "cov": None}
        self.showDisplay = True
        self.picpMaxIter = 40
        self.errorConvThresh = 1e-6
        self.LM_maxIter = 100
        self.tau = 1
        self.e1 = 1e-8
        self.e2 = 1e-8
        self.chi2_p = 0.05

    def initializeData(self):
        pass

    def computeOmega(self, q):
        pass

    ''' Point of the second cloud after being applied the transformation q ie n_i = q + c_i'''
    @abstractmethod
    def compute_nPts(self, q):
        pass

    def costFunction(self, q):
        fval = np.einsum('ki,ki', self.errors_mean, np.einsum('kij,kj->ki', self.errors_cov_inv, self.errors_mean))
        return np.sqrt(fval)

    def costFunctionJacobian(self, q, fval):
        n = len(q["pose_mean"])
        return math_utility.numericalJacobian_pdf(self.costFunction, 1, q, np.full((n,), 1e-5))

    def compute_errors(self, q):
        n_pts_array = self.compute_nPts(q)
        self.errors_mean = n_pts_array["mean"] - self.a_pts_array_assoc["mean"]
        self.errors_cov_inv = np.linalg.inv(self.a_pts_array_assoc["cov"] + n_pts_array["cov"])

    @abstractmethod
    def updateForLM(self, q):
        pass

    @abstractmethod
    def incrementLM(self, q):
        pass

    def pointToPlaneAssociations(self, nPts, assocIdxs):
        """ Project the points of the second cloud onto the tangent planes at point in the first cloud """
        vecs = self.a_pts_array_assoc - nPts
        dot = np.einsum('ij,ij->i', vecs, self.normalsInFirstScan_refFrame["mean"][assocIdxs])
        multiply = np.multiply(self.normalsInFirstScan_refFrame["mean"][assocIdxs], dot[:, np.newaxis])
        return nPts + multiply

    def LevenbergMarquardt(self, q0):
        q = q0
        n = len(q0["pose_mean"])
        identity = np.eye(n)

        self.initializeData()
        self.a_pts_array_current = self.a_pts_array
        self.updateForLM(q)

        fval = self.costFunction(q)
        J_f = self.costFunctionJacobian(q, fval)

        Fval = np.linalg.norm(fval) ** 2

        J_fT = np.transpose(J_f)
        H = np.matmul(J_fT, J_f)
        g = fval * J_fT

        ''' Check g norm '''
        hasConverged = (np.linalg.norm(g, np.inf) <= self.e1)
        #if (hasConverged):
           #print(" Ending condition : norm_inf(g) = {} <= e1".format(g))

        lambda_val = self.tau * np.amax(np.diagonal(H))
        iter = 0
        v = 2.

        ''' Keep data at each iterations '''
        path = {'time': [0], 'pose': [q], 'cost': [Fval]}

        while (not hasConverged and iter < self.LM_maxIter):
            start = time.time()
            iter += 1
            H_inv = np.linalg.inv(H + lambda_val * identity)
            epsilon_mean = -np.matmul(H_inv, g).flatten()
            epsilon_mean_norm = np.linalg.norm(epsilon_mean)
            if (epsilon_mean_norm < (self.e2 ** 2)):
                hasConverged = True
                #print("Ending condition on epsilon norm: {} < {}".format(epsilon_mean, self.e2 ** 2))
            else:
                ''' Increment '''
                qnew = self.incrementLM(q, epsilon_mean)

                self.updateForLM(qnew)

                fval = self.costFunction(qnew)
                Fval_new = np.linalg.norm(fval) ** 2

                tmp = lambda_val * epsilon_mean - g.flatten()
                denom = 0.5 * np.dot(tmp.flatten(), epsilon_mean)
                l = (Fval - Fval_new) / denom
                if (l > 0):
                    q = qnew
                    Fval = Fval_new

                    J_f = self.costFunctionJacobian(q, fval)
                    J_fT = np.transpose(J_f)
                    H = np.matmul(J_fT, J_f)
                    g = fval * J_fT

                    ''' Check g norm '''
                    norm_g = np.linalg.norm(g, np.inf)
                    hasConverged = (norm_g <= self.e1)
                    #if (hasConverged):
                    #    print(" Ending condition : norm_inf(g) = {} <= e1".format(g))

                    lambda_val *= np.max([0.33, 1. - (2. * l - 1.) ** 3.])
                    v = 2.
                    compTime = time.time() - start
                    path['time'].append(path['time'][len(path['time']) - 1] + compTime)
                    path['cost'].append(Fval)
                    path['pose'].append(q)
                else:
                    lambda_val *= v
                    v *= 2.

        # print("-- LM finished --")
        # print("Final sqr error : {}".format(Fval))
        # print("Iterations : {}".format(iter))
        # print("Final pose : {}".format(x))
        return q, Fval, path

    def displayCurrentState(self, n_pts, assoc_index_N, n_gt_pts, normals=None):
        if(self.showDisplay):
            self.viewer.display(self.a_pts_array, self.c_pts_array, n_pts,
                                n_gt_pts, None, self.a_pts_array_assoc, assoc_index_N, normals)

    def PICP(self, q0, q_gt_mean=None):
        if(self.showDisplay):
            self.viewer = PICP_display.PICP_display(showAssociations = True)
            self.viewer.associationType = self.associationType
        associationThreshold = scipy.stats.chi2.ppf(1. - self.chi2_p, df=self.dim)

        n_gt_points = None
        if (q_gt_mean is not None):
            n_gt_points = poses_euler.composePoseEulerPoint(q_gt_mean, self.c_pts_array["mean"])
        q = q0

        path = {'time': [0], 'pose': [q0], 'cost': [100000]}
        iter = 0
        hasFinished = False
        while (not hasFinished):
            start = time.time()
            iter += 1

            self.c_pts_array_assoc = self.c_pts_array
            self.a_pts_array_assoc = self.a_pts_array
            self.initializeData()
            self.computeOmega(q)
            n_pts = self.compute_nPts(q)

            self.displayCurrentState(n_pts, None, n_gt_points)

            idxs_pointsA, idxs_pointsN, indiv_compat_A = dataAssociation.dataAssociation(
                dataAssociation.mahalanobisMetric,
                self.a_pts_array, n_pts,
                associationThreshold)

            # No association ?
            if(idxs_pointsN.shape[0] == 0):
                print("No association --> Stop PICP")
                hasFinished = True
                break

            self.c_pts_array_assoc = {"mean": self.c_pts_array["mean"][idxs_pointsN], "cov": self.c_pts_array["cov"][idxs_pointsN]}
            self.a_pts_array_assoc = {"mean": self.a_pts_array["mean"][idxs_pointsA],
                                      "cov": self.a_pts_array["cov"][idxs_pointsA]}

            if(self.associationType == "pointToPlane"):
                self.a_pts_array_assoc = self.pointToPlaneAssociations(self.a_pts_array_assoc["mean"],
                                                                       n_pts["mean"][idxs_pointsN],
                                                                       n_pts["cov"][idxs_pointsN],
                                                                       self.normalsInFirstScan_refFrame["mean"][idxs_pointsA, :],
                                                                       self.normalsInFirstScan_refFrame["cov"][idxs_pointsA, :, :]
                                                                               )

            q, error, _ = self.LevenbergMarquardt(q)

            if (self.associationType == "pointToPlane"):
                self.displayCurrentState(n_pts, idxs_pointsN, n_gt_points, self.normalsInFirstScan_refFrame["mean"][idxs_pointsA, :])
            else:
                self.displayCurrentState(n_pts, idxs_pointsN, n_gt_points)

            compTime = time.time() - start
            path['time'].append(path['time'][len(path['time']) - 1] + compTime)
            path['cost'].append(error)
            path['pose'].append(q)

            if (error < self.errorConvThresh or iter == self.picpMaxIter):
                hasFinished = True

        return q, path

class gaussianPICP_2D(gaussianPICP):
    def __init__(self, a_pts_array, c_pts_array):
        gaussianPICP.__init__(self, a_pts_array, c_pts_array)
        self.dim = 2

    def compute_nPts(self, q):
        return poses2D.composePosePDFPoint_array(q, self.c_pts_array_assoc)

    def costFunction(self, q):
        self.updateForLM(q)
        return super().costFunction(q)

    def updateForLM(self, q):
        self.compute_errors(q)

    def incrementLM(self, q, eps):
        return {"pose_mean": poses2D.composePose(q["pose_mean"], eps),
                "pose_cov": q["pose_cov"]}

class gaussianPICP_Euler(gaussianPICP):
    def __init__(self, a_pts_array, c_pts_array):
        gaussianPICP.__init__(self, a_pts_array, c_pts_array)
        self.dim = 3

    def compute_nPts(self, q):
        return poses_euler.composePosePDFEulerPoint_array(q, self.c_pts_array_assoc)

    def costFunction(self, q):
        self.updateForLM(q)
        return super().costFunction(q)

    def updateForLM(self, q):
        self.compute_errors(q)

    def incrementLM(self, q, eps):
        return {"pose_mean": poses_euler.composePoseEuler(q["pose_mean"], eps),
                "pose_cov": q["pose_cov"]}

class gaussianPICP_Quaternion(gaussianPICP):
    def __init__(self,a_pts_array, c_pts_array):
        gaussianPICP.__init__(self,a_pts_array, c_pts_array)
        self.dim = 3

    def compute_nPts(self, q):
        return poses_quat.composePosePDFQuatPoint(q, self.c_pts_array_assoc)

    def costFunction(self, q):
        q_mean_norm = q["pose_mean"].copy()
        q_mean_norm[0:4] /= np.linalg.norm(q["pose_mean"][0:4])
        q_norm = {"pose_mean": q_mean_norm, "pose_cov": q["pose_cov"]}
        self.updateForLM(q_norm)
        return super().costFunction(q_norm)

    def updateForLM(self, q):
        self.compute_errors(q)

    def incrementLM(self, q, eps):
        qnew_mean = q["pose_mean"] + eps
        norm = np.linalg.norm(qnew_mean[0:4])
        qnew_mean[0:4] /= norm
        return {"pose_mean": qnew_mean, "pose_cov": q["pose_cov"]}

class gaussianPICP_se(gaussianPICP):
    def __init__(self,a_pts_array, c_pts_array):
        gaussianPICP.__init__(self, a_pts_array, c_pts_array)
        self.dim = 3
        self.C_x = None
        self.U = None
        self.K = None
        self.omega = None
        self.R = None

    def initializeData(self):
        self.compute_C_x()
        self.compute_U()

    def compute_C_x(self):
        self.C_x = np.zeros((self.c_pts_array_assoc["mean"].shape[0], 3, 3))
        math_utility.vectToSkewMatrixArray(self.c_pts_array_assoc["mean"], self.C_x)

    def compute_U(self):
        n = self.C_x.shape[0]
        self.U = np.zeros((n, 3, 6))
        ones = np.full(n, 1.)
        self.U[:, :, 0:3] = -self.C_x
        self.U[:, 0, 3] = ones
        self.U[:, 1, 4] = ones
        self.U[:, 2, 5] = ones

    def compute_nPts(self, q):
        return {"mean": poses_euler.composePoseEulerPoint(q["pose_mean"], self.c_pts_array_assoc["mean"]),
                "cov": np.einsum('ij,kjl->kil', self.R, np.einsum('kij,jl->kil', self.omega, self.R.T))}

    def computeOmega(self, q):
        self.R = scipy.spatial.transform.Rotation.from_euler('ZYX', q["pose_mean"][0:3]).as_matrix()

        ''' Omega_i  = Sigma_c_i + Sigma_q_22 - [c_i]_x Sigma_q_11 [c_i]_x'''
        A_ = np.einsum('ij,kjl->kil', q["pose_cov"][0:3, 0:3], self.C_x)
        A = A_ + q["pose_cov"][0:3, 3:6]
        B = np.einsum('kij, kjl-> kil', self.C_x, A)
        C = np.einsum('ij,kjl->kil', q["pose_cov"][3:6, 0:3], self.C_x)
        self.omega = self.c_pts_array_assoc["cov"] + q["pose_cov"][3:6, 3:6] - B + C

    def computeK(self):
        ''' Sigma_ei_inv * R '''
        self.K = np.einsum('kij,jl->kil', self.errors_cov_inv, self.R)

    def updateForLM(self, q):
        self.computeOmega(q)
        self.compute_errors(q)
        self.computeK()

    ''' Jacobian of function cost taking in account variation sigma_e '''

    def costFunctionJacobian(self, q, fval):
        n = self.errors_mean.shape[0]
        J = np.empty((1, 6))
        J[:, 3:6] = 2. * np.einsum('kj,kjl->l', self.errors_mean, self.K)[None, :]
        eT_K = np.einsum('kj, kjl->kl', self.errors_mean, self.K)
        A = np.zeros((n, 3, 3))
        for i in range(0, 3):
            A_ = np.einsum('kij,jl->kil', self.omega, G[i])
            A_T = np.transpose(A_, (0, 2, 1))
            A[:, :, i] = np.einsum('kji,ki->kj', A_ + A_T, eT_K)
        C = A - 2. * self.C_x
        J[:, 0:3] = np.sum(np.einsum('kj,kjl->kl', eT_K, C), axis=0)
        return J / (2. * fval)

    def incrementLM(self, q, eps):
        exp_epsilon_R, exp_epsilon_t = math_utility.exp_SE3(eps)

        qincr_mean = np.concatenate(
            (scipy.spatial.transform.Rotation.from_dcm(exp_epsilon_R).as_euler('ZYX'), exp_epsilon_t.flatten()))

        return {'pose_mean': poses_euler.composePoseEuler(q['pose_mean'], qincr_mean),
                'pose_cov': q['pose_cov']}

    def hessianCost_q(self, w_hat, xivec):
        w_hat_sqr = w_hat @ w_hat

        Lambda_w_ = np.einsum('kij,jl->kil', self.omega, w_hat)
        Lambda_w = Lambda_w_ + np.transpose(Lambda_w_, (0, 2, 1))

        K_w = np.einsum('kij, klj->kil', Lambda_w, self.K)
        K_w_e = np.einsum('kij,kj->ki', K_w, self.errors_mean)

        D_w = 2 * np.einsum('ij,kjl->kil', w_hat, Lambda_w_) - np.einsum('kij,jl->kil', self.omega, w_hat_sqr) - np.einsum(
            'ij,kjl->kil', w_hat_sqr, self.omega)

        eT_K = np.einsum('kj, kjl->kl', self.errors_mean, self.K)

        U_xi = np.einsum('kij,j->ki', self.U, xivec)
        RU_xi = np.einsum('ij,kj->ki', self.R, U_xi)

        w_c_tau = np.einsum('kij, j->ki', self.U[:, :, 0:3], xivec[0:3]) + xivec[3:6]
        w_w_c_tau = np.einsum('ij,kj->ki', w_hat, w_c_tau)
        D_wKT = np.einsum('kij, klj->kil', D_w, self.K)
        D_wKTe = np.einsum('kij,kj->ki', D_wKT, self.errors_mean)

        K_wR = np.einsum('kij,jl->kil', K_w, self.R)
        v_3 = np.einsum('kij,kj->ki', K_wR, 2. * K_w_e + 3. * U_xi)

        A = np.einsum('kj,kj->k', eT_K, 2. * w_w_c_tau + D_wKTe + v_3)

        RU_xiT_K = np.einsum('kj,kjl->kl', RU_xi, self.K)
        B = np.einsum('kj,kj->k', RU_xiT_K, 2. * U_xi + K_w_e)

        return A + B

    def hessianMatrixCost_z(self, q):
        n = self.errors_mean.shape[0]
        H_z = np.zeros((n, 6, 6))

        A = np.einsum('kji,kjl->kil', self.U, np.einsum('kji,jl->kil', self.K, self.R))
        B = np.zeros((n, 6, 3))
        dU = np.zeros((3, 6))
        for i in range(0, 6):
            xivec = np.zeros((6,))
            xivec[i] = 1.
            Lambda_w_ = np.einsum('kij,jl->kil', self.omega, G[i])
            Lambda_w = Lambda_w_ + np.transpose(Lambda_w_, (0, 2, 1))

            K_w = np.einsum('kij, klj->kil', Lambda_w, self.K)
            eT_KwT = np.einsum('kj, kij ->ki', self.errors_mean, K_w)
            K_Kw = np.einsum('kij, kjl->kil', self.K, K_w)
            xivec_UT = np.einsum('j,kij->ki', xivec, self.U)

            ''' Derivative w.r.t to points a'''
            H_z[:, i, 0:3] = -(
                        np.einsum('kj,kij->ki', 2. * xivec_UT + eT_KwT, self.K) + np.einsum('kj,klj->kl', self.errors_mean, K_Kw))

            ''' Derivative w.r.t to points c'''
            KU = np.einsum('kij,kjl->kil', self.K, self.U)
            K_Kw_T = np.transpose(K_Kw, (0, 2, 1))
            B[:, i, :] = np.einsum('kj,kji->ki', self.errors_mean, np.einsum('kij,jl->kil', K_Kw + K_Kw_T, self.R))
            for j in range(0, 3):
                dU[:, 0:3] = -G[j]
                K1 = np.einsum('ij,kjl->kil', dU, np.einsum('ij,klj->kil', q["pose_cov"], self.U))
                K2 = np.einsum('kij,jl->kil', self.U, np.einsum('ij, lj->il', q["pose_cov"], dU))
                dOmega = K1 + K2
                dSigma = np.einsum('ij,kjl->kil', self.R, np.einsum('kij,jl->kil', dOmega, self.R.T))
                sigmaInv_dSigma = np.einsum('kij,kjl->kil', self.errors_cov_inv, dSigma)

                dKU = np.einsum('kij, jl->kil', self.K, dU) - np.einsum('kij,kjl->kil', sigmaInv_dSigma, KU)
                A_ = np.einsum('...j,...j', self.errors_mean, dKU[:, :, i])
                A[:, i, j] += A_

                K_Lambda_w = np.einsum('kij,kjl->kil', self.K, Lambda_w)
                dLambda_w_ = np.einsum('kij,jl->kil', dOmega, G[i])
                dLambda_w = dLambda_w_ + np.transpose(dLambda_w_, (0, 2, 1))
                dKwT = -np.einsum('kij,kjl->kil', sigmaInv_dSigma, K_Lambda_w) + np.einsum('kij,kjl->kil', self.K, dLambda_w)
                dKwT_RT_sigmaInv = np.einsum('kij,klj->kil', dKwT, self.K) - np.einsum('kji,kjl->kil', K_Kw,
                                                                                  np.einsum('kij,kjl->kil', dSigma,
                                                                                            self.errors_cov_inv))
                B[:, i, j] += np.einsum('...j,...j', self.errors_mean,
                                        np.einsum('kij, kj->ki', dKwT_RT_sigmaInv, self.errors_mean))

        H_z[:, :, 3:6] = 2. * A + B

        return H_z

    ''' Hessian w.r.t q (ie H_q in paper) of cost function'''
    def hessianMatrixCost_q(self):
        n = self.errors_mean.shape[0]
        self.H_q = np.empty((n, 6, 6))

        ''' Compute diagonal elements first with simplified expressions '''
        for i in range(0, 6):
            xivec = np.zeros((6,))
            xivec[i] = 1.
            self.H_q[:, i, i] = self.hessianCost_q(G[i], xivec)

        ''' Off diagonal elements with full expressions '''
        for i in range(0, 6):
            for j in range(0, i):
                xivec = np.zeros((6,))
                xivec[i] = 1.
                xivec[j] = 1.
                self.H_q[:, i, j] = 0.5 * (
                            self.hessianCost_q(G[i] + G[j], xivec) - self.H_q[:, i, i] - self.H_q[:, j,j])
                self.H_q[:, j, i] = self.H_q[:, i, j]

        return np.sum(self.H_q, axis=0)

    def hessianMatrixCost(self, q):
        self.H_q = self.hessianMatrixCost_q()
        self.H_z = self.hessianMatrixCost_z(q)

    def LM_covariance_(self, Sigma_z):
        H_q_inv = np.linalg.inv(self.H_q)

        ''' Sum_i H_q,zi Sigma_zi H_q,zi^T '''
        A = np.sum(np.einsum('kij, kjl->kil', self.H_z, np.einsum("kij,klj->kil", Sigma_z, self.H_z)), axis=0)
        cov = H_q_inv @ A @ H_q_inv

        return cov

    def LM_covariance(self, q):
        self.hessianMatrixCost(q)

        Sigma_z = np.zeros((self.H_z.shape[0], 6, 6))
        Sigma_z[:, 0:3, 0:3] = self.a_pts_array_assoc["cov"]
        Sigma_z[:, 3:6, 3:6] = self.c_pts_array_assoc["cov"]

        return self.LM_covariance_(Sigma_z)

    def jacobianProjectionToPlane_wrt_n(self, normal_means):
        # dai_dni = I_3 - vi vi^T , ai = firstCloudPt, ni= secondCloudPt after composition, vi = normal mean
        return np.eye(3) - np.einsum("ki,kj->kij", normal_means, normal_means)

    def jacobianNormalNormalization(self, normal_means):
        nx_sqr = np.power(normal_means[:,0], 2)
        ny_sqr = np.power(normal_means[:, 1], 2)
        nz_sqr = np.power(normal_means[:, 2], 2)
        nx_ny = normal_means[:,0] * normal_means[:,1]
        nx_nz = normal_means[:,0] * normal_means[:,2]
        ny_nz = normal_means[:,1] * normal_means[:,2]

        K = 1./np.power(np.linalg.norm(normal_means[:,0:4],axis=1), 3)
        jacobian_normalization = np.empty((normal_means.shape[0], 3, 3))
        jacobian_normalization[:,0,0] = ny_sqr + nz_sqr
        jacobian_normalization[:, 0, 1] = -nx_ny
        jacobian_normalization[:, 0, 2] = -nx_nz
        jacobian_normalization[:, 1, 0] = -nx_ny
        jacobian_normalization[:, 1, 1] = nx_sqr + nz_sqr
        jacobian_normalization[:, 1, 2] = -ny_nz
        jacobian_normalization[:, 2, 0] = -nx_nz
        jacobian_normalization[:, 2, 1] = -ny_nz
        jacobian_normalization[:, 2, 2] = nx_sqr + ny_sqr

        return K[:,None,None]*jacobian_normalization

    def jacobianProjectionToPlane_wrt_normal(self, normal_means, vecs, dot):

        # dai_dvi = vi(ni - ai)^T + vi^T(ni - ai)I_3 , here dot = vi^T(ni -ai) and vecs = ni -ai
        # (vi unnormalized)
        # The final jacobian is dai_dvi * dvi_norm_dvi (jacobian_norm)
        A = np.zeros((normal_means.shape[0],3,3))
        A[:,0,0] = dot
        A[:,1,1] = dot
        A[:,2,2] = dot

        J = -(np.einsum("ki,kj->kij", normal_means, vecs) + A)

        jacobian_norm = self.jacobianNormalNormalization(normal_means)

        return np.einsum("kij,kjl->kil", J, jacobian_norm)

    def projectPointToTangentPlanes(self,firstCloud_means, secondCloud_means, secondCloud_covs, normal_means, normal_covs):
        vecs = secondCloud_means - firstCloud_means
        dot = np.einsum('ij,ij->i', vecs, normal_means)
        multiply = np.multiply(normal_means, dot[:, np.newaxis])

        jacobian_n = self.jacobianProjectionToPlane_wrt_n(normal_means)
        jacobian_v = self.jacobianProjectionToPlane_wrt_normal(normal_means,
                                                          vecs,
                                                          dot)
        cov = np.einsum("kij,kjl->kil", jacobian_n, np.einsum("kij,klj->kil", secondCloud_covs, jacobian_n)) + \
              np.einsum("kij,kjl->kil", jacobian_v, np.einsum("kij,klj->kil", normal_covs, jacobian_v))

        return {"mean": secondCloud_means - multiply, "cov": cov}

    def pointToPlaneAssociations(self,firstCloud_means, secondCloud_means, seconCloud_covs, normal_means, normal_covs):
        """ Project the points of the second cloud onto the tangent planes at point in the first cloud """
        projectedPointsPDF = self.projectPointToTangentPlanes(firstCloud_means, secondCloud_means, seconCloud_covs, normal_means, normal_covs )
        return projectedPointsPDF

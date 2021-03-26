''' Operations and functions related to SE(3) poses
'''

import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats
import time
import poses_euler
import multiprocessing as mp

''' g '''
def fromSphericalToCart(x):
    cos_theta = np.cos(x[1])
    sin_theta = np.sin(x[1])
    cos_phi = np.cos(x[2])
    sin_phi = np.sin(x[2])

    return np.array([x[0] * cos_theta * cos_phi, x[0] * cos_theta * sin_phi, x[0] * sin_theta])

def fromSphericalToCart_array(x_array):
    n = x_array.shape[0]
    cos_thetas = np.cos(x_array[:,1])
    sin_thetas = np.sin(x_array[:,1])
    cos_phis = np.cos(x_array[:,2])
    sin_phis = np.sin(x_array[:, 2])

    res = np.empty((n,3))
    res[:,0] = x_array[:,0] * cos_thetas * cos_phis
    res[:,1] = x_array[:,0] * cos_thetas * sin_phis
    res[:,2] = x_array[:,0] * sin_thetas
    return res

''' g^-1 '''
def fromCartToSpherical(x):
    rho = np.linalg.norm(x)
    return np.array([rho, np.arcsin(x[2] / rho), np.arctan2(x[1], x[0])])

''' Generate a skew matrix from a 3D vector '''
def vectToSkewMatrix(x):
    return np.array([[0., -x[2], x[1]],
                    [x[2], 0., -x[0]],
                    [-x[1], x[0], 0.]], dtype=float)

def vectToSkewMatrixArray(x_array, res):
    #TODO : Faster way to do it ?
    res[:,0,1] = -x_array[:,2]
    res[:,0,2] = x_array[:,1]
    res[:,1,0] = x_array[:,2]
    res[:,1,2] = -x_array[:,0]
    res[:,2,0] = -x_array[:,1]
    res[:,2,1] = x_array[:,0]

''' Exponential map for SE(3) '''
def exp_SE3(epsilon):
    I_3 = np.eye(3)
    w = epsilon[0:3]
    tau = epsilon[3:6]
    norm_w = np.linalg.norm(w)
    if(norm_w < 1e-8):
        R = I_3
        t = tau
    else:
        norm_w_sqr_inv = 1. / norm_w ** 2
        sin = np.sin(norm_w)
        a = sin/norm_w
        b = (1 - np.cos(norm_w))*norm_w_sqr_inv
        c = (norm_w - sin)/(norm_w**3)

        w_x = vectToSkewMatrix(w)
        w_x_sqr = np.matmul(w_x,w_x)

        R = I_3 + a*w_x + b*w_x_sqr

        V = I_3 + b*w_x + c*w_x_sqr
        t = np.matmul(V,tau)

    return R,t

def exp_SE3_euler(epsilon):
    R, t = exp_SE3(epsilon)
    q_euler = np.empty((6,))
    q_euler[0:3] = scipy.spatial.transform.Rotation.from_dcm(R).as_euler("ZYX")
    q_euler[3:6] = t
    return q_euler

''' Logarithm map for SO(3) '''
def log_SO3(p):
    q = scipy.spatial.transform.Rotation.from_euler('ZYX',p).as_quat()
    squared_n = q[0]*q[0] + q[1]*q[1] + q[2]*q[2]
    n = np.sqrt(squared_n)
    if (n < 1e-7):
        two_atan = 2./q[3] - 2.*squared_n/(q[3]**3)
    elif (np.abs(q[3]) < 1e-7):
        if q[3] > 0:
            two_atan = np.pi / n
        else:
            two_atan = -np.pi / n
    else:
        two_atan = 2.*np.arctan(n / q[3]) / n

    return two_atan*q[0:3]

''' Logarithm map for SE(3) '''
def log_SE3(p):
    log_R = log_SO3(p[0:3])

    norm_w = np.linalg.norm(log_R)
    norm_w_sqr_inv = 1. / norm_w ** 2
    sin = np.sin(norm_w)
    w_x = vectToSkewMatrix(log_R)
    w_x_sqr = np.matmul(w_x, w_x)
    b = (1 - np.cos(norm_w)) * norm_w_sqr_inv
    c = (norm_w - sin) / (norm_w ** 3)
    V = np.eye(3) + b * w_x + c * w_x_sqr

    return np.concatenate((log_R, np.linalg.solve(V,p[3:6]))) #np.linalg.inv(V)@p[3:6]))

def distanceSE3(p1, p2):
    G = np.block([[2.*np.eye(3), np.zeros((3,3))],
                  [np.zeros((3,3)), np.eye(3)]])
    log = log_SE3(poses_euler.inverseComposePoseEuler(p1, p2))
    return np.sqrt(np.dot(log, G@log))

''' Numerically compute a jacobian '''
''' Used principally to test the closed form jacobian '''
def numericalJacobian(func, output_dim, x, increments, *args, **kargs):
    i = 0
    m = len(x)
    jacobian = np.zeros((output_dim, m))
    for x_i, incr_i in zip(x, increments):
        x_mod = x.copy()
        x_mod[i] = x_i + incr_i
        f_plus = func(x_mod, *args, **kargs)

        x_mod[i] = x_i - incr_i
        f_minus = func(x_mod, *args, **kargs)

        denum = 0.5/incr_i

        #print("f_plus, f_minus : {},{}".format(f_plus, f_minus))

        if(output_dim == 1):
            jacobian[0][i] = denum*(f_plus - f_minus)
        else:
            for j in range(0, output_dim):
                jacobian[j][i] = denum*(f_plus[j] - f_minus[j])

        i = i+1

    return jacobian

def numericalJacobian_pool(func, output_dim, x, increments, *args, **kargs):
    pool = mp.Pool(mp.cpu_count())

    i = 0
    m = len(x)
    jacobian = np.zeros((output_dim, m))
    args_list = []
    denum = []
    for x_i, incr_i in zip(x, increments):
        x_mod = x.copy()
        x_mod[i] = x_i + incr_i
        args_list.append(x_mod)

    for x_i, incr_i in zip(x, increments):
        x_mod = x.copy()
        x_mod[i] = x_i - incr_i
        args_list.append(x_mod)
        denum.append(0.5/incr_i)

        #print("f_plus, f_minus : {},{}".format(f_plus, f_minus))

    func_vals = pool.map(func, args_list)
    pool.close()
    pool.join()

    jacobian[0] = denum * (func_vals[:m] - func_vals[m:])

    '''if(output_dim == 1):
        jacobian[0] = denum*(func_vals[:m] - func_vals[m:])
    else:
        jacobian
        for j in range(0, output_dim):
            jacobian[j][i] = denum*(f_plus[j] - f_minus[j])'''

    return jacobian

''' Numerically compute a jacobian '''
''' Used principally to test the closed form jacobian '''
def numericalJacobian_pdf(func, output_dim, x, increments):
    i = 0
    m = len(x["pose_mean"])
    jacobian = np.zeros((output_dim, m))
    x_mod = {"pose_mean": None, "pose_cov":x["pose_cov"]}
    for x_i, incr_i in zip(x["pose_mean"], increments):
        x_mod["pose_mean"] = x["pose_mean"].copy()
        x_mod["pose_mean"][i] = x_i + incr_i
        f_plus = func(x_mod)

        x_mod["pose_mean"][i] = x_i - incr_i
        f_minus = func(x_mod)

        denum = 0.5/incr_i

        #print("f_plus, f_minus : {},{}".format(f_plus, f_minus))

        if(output_dim == 1):
            jacobian[0][i] = denum*(f_plus - f_minus)
        else:
            for j in range(0, output_dim):
                jacobian[j][i] = denum*(f_plus[j] - f_minus[j])

        i = i+1

    return jacobian

def numericalHessian(func, output_dim, x, increments, *args):
    output_dim_jacobian = len(x)
    return numericalJacobian(lambda y: numericalJacobian(func, output_dim, y, increments, *args).T, output_dim_jacobian,
                             x, increments, *args)

def LevenbergMarquardt(x0, maxIter, updateArgsFunc, func, jacobFunc, incrFunc,
                       n, tau=1., e1=1e-8, e2=1e-8, **kargsInit):
    x = x0

    identity = np.eye(n)

    kargs = updateArgsFunc(x, **kargsInit)

    fval = func(x, **kargs)# **kargsInit)
    J_f = jacobFunc(x, fval, **kargs)#, **kargsInit)

    Fval = np.linalg.norm(fval)**2

    #print("First Fval : {}".format(Fval))

    J_fT = np.transpose(J_f)
    H = np.matmul(J_fT, J_f)
    if(isinstance(fval,float)):
        g = fval * J_fT
    else:
        g = J_fT@fval

    ''' Check g norm '''
    hasConverged = (np.linalg.norm(g, np.inf) <= e1)
    if (hasConverged):
        print(" Ending condition : norm_inf(g) = {} <= e1".format(g))

    lambda_val = tau * np.amax(np.diagonal(H))
    iter = 0
    v = 2.

    ''' Keep data at each iterations '''
    path = {'time': [0], 'pose': [x], 'cost': [Fval]}

    while (not hasConverged and iter < maxIter):
        start = time.time()
        iter += 1
        H_inv = np.linalg.inv(H + lambda_val * identity)
        epsilon_mean = -np.matmul(H_inv, g).flatten()
        epsilon_mean_norm = np.linalg.norm(epsilon_mean)
        epsilon = {"pose_mean": epsilon_mean, "pose_cov":  H_inv}
        if (epsilon_mean_norm < (e2 ** 2)):
            hasConverged = True
            print("Ending condition on epsilon norm: {} < {}".format(epsilon_mean, e2 ** 2))
        else:
            ''' Increment '''
            xnew = incrFunc(x, epsilon, **kargs)#, **kargsInit)
            kargs = updateArgsFunc(xnew, **kargsInit)

            fval = func(xnew, **kargs)#, **kargsInit)
            Fval_new = np.linalg.norm(fval)**2

            tmp = lambda_val * epsilon["pose_mean"] - g.flatten()
            denom = 0.5 * np.dot(tmp.flatten(), epsilon["pose_mean"])
            l = (Fval - Fval_new) / denom
            if (l > 0):
                # print("Accept increment")
                x = xnew

                Fval = Fval_new
                #print("Current Fval : {}".format(Fval))
                #print("Current fval : {}".format(fval))

                #print("xnew_math : {}".format(xnew))
                J_f = jacobFunc(x, fval, **kargs)#, **kargsInit)
                #print("J_f_math : {}".format(J_f))
                #print("fval_math : {}".format(fval))
                J_fT = np.transpose(J_f)
                H = np.matmul(J_fT, J_f)
                if (isinstance(fval, float)):
                    g = fval * J_fT
                else:
                    g = J_fT @ fval

                ''' Check g norm '''
                norm_g = np.linalg.norm(g, np.inf)
                hasConverged = (norm_g <= e1)
                if (hasConverged):
                    print(" Ending condition : norm_inf(g) = {} <= e1".format(g))

                lambda_val *= np.max([0.33, 1. - (2. * l - 1.) ** 3.])
                v = 2.
                compTime = time.time() - start
                path['time'].append(path['time'][len(path['time']) - 1] + compTime)
                path['cost'].append(Fval)
                path['pose'].append(x)
            else:
                lambda_val *= v
                v *= 2.

   # print("-- LM finished --")
   # print("Final sqr error : {}".format(Fval))
   # print("Iterations : {}".format(iter))
   # print("Final pose : {}".format(x))
    return x, Fval, path